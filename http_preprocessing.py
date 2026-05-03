import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Iterable

import torch


BOT_LABELS = {"bot", "bots", "abnormal", "malicious", "1", 1, True}
HUMAN_LABELS = {"human", "humano", "unsafe", "normal", "0", 0, False}


def parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, dict) and "$date" in value:
        return parse_datetime(value["$date"])
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value)
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    raise ValueError(f"Unsupported datetime value: {value!r}")


def coerce_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {"_raw": value}
        if isinstance(parsed, dict):
            return parsed
        return {"_raw": parsed}
    return {"_raw": value}


def normalize_label(value: Any) -> int:
    if value in BOT_LABELS:
        return 1
    if value in HUMAN_LABELS:
        return 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in BOT_LABELS:
            return 1
        if lowered in HUMAN_LABELS:
            return 0
    raise ValueError(f"Unknown decision label: {value!r}")


def stable_hash(text: str, buckets: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = defaultdict(int)
    for char in text:
        counts[char] += 1
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def iter_kv_tokens(prefix: str, mapping: dict[str, Any]) -> Iterable[str]:
    for key, value in sorted((mapping or {}).items()):
        key_text = str(key).strip().lower()
        value_text = str(value).strip().lower()
        yield f"{prefix}:key:{key_text}"
        yield f"{prefix}:pair:{key_text}={value_text}"
        if value_text:
            yield f"{prefix}:value:{value_text}"


def event_to_text(event: dict[str, Any]) -> str:
    headers = coerce_mapping(event.get("headers"))
    request = coerce_mapping(event.get("request"))

    def render_mapping(name: str, mapping: dict[str, Any]) -> str:
        parts = [f"{key}={value}" for key, value in sorted(mapping.items())]
        return f"{name}: " + " ".join(parts)

    return " | ".join(
        [
            f"ip: {extract_ip(event)}",
            render_mapping("headers", headers),
            render_mapping("request", request),
        ]
    )


def event_to_feature(event: dict[str, Any], feature_dim: int = 64) -> torch.Tensor:
    if feature_dim < 12:
        raise ValueError("feature_dim must be at least 12")

    headers = coerce_mapping(event.get("headers"))
    request = coerce_mapping(event.get("request"))
    hash_dim = feature_dim - 8
    vector = torch.zeros(feature_dim, dtype=torch.float32)

    for token in iter_kv_tokens("h", headers):
        vector[stable_hash(token, hash_dim)] += 1.0
    for token in iter_kv_tokens("q", request):
        vector[stable_hash(token, hash_dim)] += 1.0

    header_values = " ".join(str(value) for value in headers.values())
    request_values = " ".join(str(value) for value in request.values())
    user_agent = str(headers.get("user-agent") or headers.get("User-Agent") or "")

    numeric = torch.tensor(
        [
            math.log1p(len(headers)),
            math.log1p(len(request)),
            math.log1p(len(user_agent)),
            shannon_entropy(user_agent) / 8.0,
            int("user-agent" not in {str(k).lower() for k in headers}),
            int(any("bot" in str(v).lower() for v in headers.values())),
            shannon_entropy(header_values) / 8.0,
            shannon_entropy(request_values) / 8.0,
        ],
        dtype=torch.float32,
    )
    vector[hash_dim:] = numeric
    vector[:hash_dim] = torch.sign(vector[:hash_dim]) * torch.log1p(torch.abs(vector[:hash_dim]))
    return vector


def extract_ip(event: dict[str, Any]) -> str:
    headers = coerce_mapping(event.get("headers"))
    ip = (
        event.get("ip")
        or event.get("remote_addr")
        or event.get("client_ip")
        or headers.get("x-forwarded-for")
        or headers.get("X-Forwarded-For")
        or headers.get("x-real-ip")
        or headers.get("X-Real-IP")
    )
    if isinstance(ip, str) and "," in ip:
        ip = ip.split(",", 1)[0].strip()
    return str(ip or "unknown-ip")


def default_session_key(event: dict[str, Any]) -> str:
    return extract_ip(event)


def preprocess_http_events(
    events: list[dict[str, Any]],
    feature_dim: int = 64,
    max_len: int | None = None,
    session_key_fn: Callable[[dict[str, Any]], str] = default_session_key,
    feature_fn: Callable[[dict[str, Any]], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    if feature_fn is None:
        feature_fn = lambda event: event_to_feature(event, feature_dim=feature_dim)

    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        sessions[session_key_fn(event)].append(event)

    ordered_session_ids = sorted(sessions)
    if max_len is None:
        max_len = max(len(sessions[session_id]) for session_id in ordered_session_ids)

    x = torch.zeros(len(ordered_session_ids), max_len, feature_dim, dtype=torch.float32)
    mask = torch.zeros(len(ordered_session_ids), max_len, dtype=torch.bool)
    y = torch.zeros(len(ordered_session_ids), dtype=torch.float32)

    for batch_idx, session_id in enumerate(ordered_session_ids):
        session = sorted(sessions[session_id], key=lambda item: parse_datetime(item["datetime"]))
        clipped = session[-max_len:]
        label_values = [normalize_label(item["decision"]) for item in clipped if "decision" in item]
        y[batch_idx] = float(max(label_values) if label_values else 0)

        previous_time = None
        deltas = []
        features = []
        for event in clipped:
            current_time = parse_datetime(event["datetime"])
            deltas.append(0.0 if previous_time is None else max((current_time - previous_time).total_seconds(), 0.0))
            previous_time = current_time
            features.append(feature_fn(event))

        session_tensor = torch.stack(features)
        delta_tensor = torch.tensor(deltas, dtype=torch.float32)
        if len(delta_tensor) > 1:
            delta_tensor = (delta_tensor - delta_tensor.mean()) / (delta_tensor.std() + 1e-5)
        session_tensor[:, -1] = delta_tensor

        length = len(clipped)
        x[batch_idx, :length] = session_tensor
        mask[batch_idx, :length] = True

    return x, y, mask, ordered_session_ids
