import argparse
import collections
import json
from pathlib import Path
from typing import Any


DEFAULT_UNSAFE_NAMES = ("tiktok-unsafe-10k.json", "unsafe-10k.json")
DEFAULT_BOT_NAMES = ("tiktok-bot-10k.json", "bot-10k.json")


def find_data_file(candidates: tuple[str, ...]) -> Path:
    project_root = Path.cwd().parent.parent
    search_roots = [
        Path.cwd(),
        Path.cwd().parent,
        project_root,
        project_root / "data",
    ]
    for root in search_roots:
        for name in candidates:
            path = root / name
            if path.exists():
                return path
    names = ", ".join(candidates)
    raise FileNotFoundError(f"Could not find any of these files near the project: {names}")


def unwrap_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "records", "events", "requests", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("JSON must be a list of events or a dict containing data/records/events/requests/items")


def load_json_events(path: Path, decision: str, limit: int | None = None) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        records = unwrap_records(json.loads(text))
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]

    events = []
    for row_idx, record in enumerate(records[:limit]):
        event = dict(record)
        event["decision"] = event.get("decision", decision)
        event["_source_decision"] = decision
        event["_row_idx"] = row_idx
        events.append(event)
    return events


def describe_split(name: str, events: list[dict[str, Any]]) -> None:
    ips = {event.get("ip") or event.get("remote_addr") or event.get("client_ip") for event in events}
    ips.discard(None)
    print(f"{name}: {len(events)} requests, {len(ips)} explicit IPs")


def describe_sessions(session_ids: list[str], mask) -> None:
    lengths = mask.sum(dim=1).tolist()
    if not lengths:
        return
    sorted_lengths = sorted(int(value) for value in lengths)
    mean_len = sum(sorted_lengths) / len(sorted_lengths)
    p95_len = sorted_lengths[int(0.95 * len(sorted_lengths)) - 1]
    singletons = sum(length == 1 for length in sorted_lengths)
    print(f"session len mean/median/p95/max: {mean_len:.2f}/{sorted_lengths[len(sorted_lengths) // 2]}/{p95_len}/{max(sorted_lengths)}")
    print(f"single-request sessions: {singletons}/{len(session_ids)}")


def build_session_key_fn(mode: str, chunk_size: int):
    from http_preprocessing import coerce_mapping, extract_ip, parse_datetime

    def hour_bucket(event: dict[str, Any]) -> str:
        return parse_datetime(event["datetime"]).strftime("%Y-%m-%d-%H")

    if mode == "ip":
        return extract_ip
    if mode == "ip-hour":
        return lambda event: f"{extract_ip(event)}|{hour_bucket(event)}"
    if mode == "ip-user-agent-hour":
        def key(event: dict[str, Any]) -> str:
            headers = coerce_mapping(event.get("headers"))
            user_agent = headers.get("User-Agent") or headers.get("user-agent") or ""
            return f"{extract_ip(event)}|{user_agent}|{hour_bucket(event)}"
        return key
    if mode == "label-chunk":
        def key(event: dict[str, Any]) -> str:
            decision = event.get("_source_decision") or event.get("decision")
            return f"{decision}|chunk-{int(event.get('_row_idx', 0)) // chunk_size}"
        return key
    raise ValueError(f"Unknown session mode: {mode}")


def print_weak_label_overlap(events: list[dict[str, Any]]) -> None:
    by_ip: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for event in events:
        by_ip[str(event.get("ip"))][event.get("decision")] += 1
    overlaps = sum(1 for counts in by_ip.values() if len(counts) > 1)
    print(f"IPs with mixed weak labels: {overlaps}/{len(by_ip)}")


def print_semantic_validation(events: list[dict[str, Any]], feature_map: dict[int, Any], embedding_dim: int = 384) -> None:
    import torch
    import torch.nn.functional as F

    labels = torch.tensor([1 if str(event.get("_source_decision") or event.get("decision")).lower() in {"bot", "bots"} else 0 for event in events])
    embeddings = torch.stack([feature_map[id(event)][:embedding_dim] for event in events]).to(torch.float32)
    embeddings = F.normalize(embeddings, dim=1)

    unsafe_embeddings = embeddings[labels == 0]
    bot_embeddings = embeddings[labels == 1]
    if len(unsafe_embeddings) == 0 or len(bot_embeddings) == 0:
        print("semantic validation skipped: both weak-label groups are required")
        return

    unsafe_center = F.normalize(unsafe_embeddings.mean(dim=0, keepdim=True), dim=1)
    bot_center = F.normalize(bot_embeddings.mean(dim=0, keepdim=True), dim=1)
    centroid_cosine = F.cosine_similarity(unsafe_center, bot_center).item()

    sim_to_unsafe = (embeddings @ unsafe_center.T).squeeze(1)
    sim_to_bot = (embeddings @ bot_center.T).squeeze(1)
    margin = sim_to_bot - sim_to_unsafe
    predicted_by_centroid = (margin > 0).long()
    centroid_agreement = (predicted_by_centroid == labels).float().mean().item()

    similarity = embeddings @ embeddings.T
    similarity.fill_diagonal_(-2.0)
    k = min(5, len(events) - 1)
    nn_idx = torch.topk(similarity, k=k, dim=1).indices
    nn_labels = labels[nn_idx]
    nn_agreement = (nn_labels == labels.unsqueeze(1)).float().mean().item()

    unsafe_suspects = ((labels == 0) & (margin > 0)).sum().item()
    bot_suspects = ((labels == 1) & (margin < 0)).sum().item()

    print(f"semantic centroid cosine unsafe/bot: {centroid_cosine:.4f}")
    print(f"semantic centroid weak-label agreement: {centroid_agreement:.4f}")
    print(f"semantic top-{k} neighbor weak-label agreement: {nn_agreement:.4f}")
    print(f"semantic suspect weak labels unsafe->bot/bot->unsafe: {unsafe_suspects}/{bot_suspects}")


def run_smoke_test(
    unsafe_path: Path,
    bot_path: Path,
    limit_per_file: int,
    feature_dim: int,
    max_len: int,
    top_k: int,
    session_mode: str,
    chunk_size: int,
    feature_mode: str,
    minilm_model: str,
    embedding_batch_size: int,
    minilm_offline: bool,
) -> None:
    from crossbatch_memory_banck import BotDetectionNet, CrossBatchMemoryBank
    from http_preprocessing import preprocess_http_events
    from mil_functions import contrastive_clustering_loss, mil_loss

    unsafe_events = load_json_events(unsafe_path, decision="unsafe", limit=limit_per_file)
    bot_events = load_json_events(bot_path, decision="bot", limit=limit_per_file)
    events = unsafe_events + bot_events

    if not events:
        raise ValueError("No events were loaded from the JSON files")

    describe_split("unsafe/human", unsafe_events)
    describe_split("bot", bot_events)
    print_weak_label_overlap(events)

    session_key_fn = build_session_key_fn(session_mode, chunk_size)
    feature_fn = None
    effective_feature_dim = feature_dim
    if feature_mode == "minilm":
        from semantic_features import MINILM_FEATURE_DIM, build_minilm_feature_map

        print(f"semantic encoder: {minilm_model}")
        feature_map = build_minilm_feature_map(
            events,
            model_name=minilm_model,
            batch_size=embedding_batch_size,
            offline=minilm_offline,
        )
        print_semantic_validation(events, feature_map)
        feature_fn = lambda event: feature_map[id(event)]
        effective_feature_dim = MINILM_FEATURE_DIM

    x, y, mask, session_ids = preprocess_http_events(
        events,
        feature_dim=effective_feature_dim,
        max_len=max_len,
        session_key_fn=session_key_fn,
        feature_fn=feature_fn,
    )

    model = BotDetectionNet(feature_dim=effective_feature_dim, embedding_dim=32)
    memory_bank = CrossBatchMemoryBank(feature_dim=32)

    scores, features = model(x, mask=mask)
    loss_mil = mil_loss(scores, y, k=top_k, mask=mask)
    loss_contrastive = contrastive_clustering_loss(
        features,
        scores,
        y,
        memory_bank,
        k=top_k,
        mask=mask,
    )
    loss = loss_mil + loss_contrastive

    valid_scores = scores[mask].detach()
    print(f"sessions: {len(session_ids)}")
    print(f"session mode: {session_mode}")
    print(f"feature mode: {feature_mode}")
    print(f"X shape: {tuple(x.shape)}")
    print(f"y shape: {tuple(y.shape)} | bot sessions: {int(y.sum().item())}")
    print(f"mask valid requests: {int(mask.sum().item())}")
    describe_sessions(session_ids, mask)
    print(f"score mean/min/max: {valid_scores.mean().item():.4f}/{valid_scores.min().item():.4f}/{valid_scores.max().item():.4f}")
    print(f"mil loss: {loss_mil.item():.4f}")
    print(f"contrastive loss: {loss_contrastive.item():.4f}")
    print(f"total loss: {loss.item():.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real WSTAD HTTP bot/human smoke test from JSON files.")
    parser.add_argument("--unsafe-json", type=Path, default=None, help="Path to tiktok-unsafe-10k.json")
    parser.add_argument("--bot-json", type=Path, default=None, help="Path to tiktok-bot-10k.json")
    parser.add_argument("--limit-per-file", type=int, default=1000, help="Max events loaded from each JSON")
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--feature-mode", choices=("hashing", "minilm"), default="hashing")
    parser.add_argument("--minilm-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--minilm-offline", action="store_true", help="Load MiniLM from local cache without network checks")
    parser.add_argument("--max-len", type=int, default=128, help="Max requests per IP session")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--session-mode",
        choices=("ip", "ip-hour", "ip-user-agent-hour", "label-chunk"),
        default="ip",
        help="How requests are grouped into weakly supervised bags.",
    )
    parser.add_argument("--chunk-size", type=int, default=32, help="Requests per bag when --session-mode label-chunk")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unsafe_path = args.unsafe_json or find_data_file(DEFAULT_UNSAFE_NAMES)
    bot_path = args.bot_json or find_data_file(DEFAULT_BOT_NAMES)

    print(f"unsafe/human file: {unsafe_path}")
    print(f"bot file: {bot_path}")

    run_smoke_test(
        unsafe_path=unsafe_path,
        bot_path=bot_path,
        limit_per_file=args.limit_per_file,
        feature_dim=args.feature_dim,
        max_len=args.max_len,
        top_k=args.top_k,
        session_mode=args.session_mode,
        chunk_size=args.chunk_size,
        feature_mode=args.feature_mode,
        minilm_model=args.minilm_model,
        embedding_batch_size=args.embedding_batch_size,
        minilm_offline=args.minilm_offline,
    )


if __name__ == "__main__":
    main()
