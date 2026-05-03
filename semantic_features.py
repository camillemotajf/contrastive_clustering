from typing import Any

import os
import torch

from http_preprocessing import event_to_feature, event_to_text


MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_EMBEDDING_DIM = 384
NUMERIC_FEATURE_DIM = 8
MINILM_FEATURE_DIM = MINILM_EMBEDDING_DIM + NUMERIC_FEATURE_DIM


def build_minilm_feature_map(
    events: list[dict[str, Any]],
    model_name: str = MINILM_MODEL_NAME,
    batch_size: int = 64,
    normalize_embeddings: bool = True,
    offline: bool = False,
) -> dict[int, torch.Tensor]:
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for --feature-mode minilm. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    model = SentenceTransformer(model_name)
    texts = [event_to_text(event) for event in events]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=True,
    ).to(torch.float32)

    numeric_features = torch.stack([event_to_feature(event, feature_dim=64)[-NUMERIC_FEATURE_DIM:] for event in events])
    features = torch.cat([embeddings.cpu(), numeric_features], dim=1)
    return {id(event): features[idx] for idx, event in enumerate(events)}
