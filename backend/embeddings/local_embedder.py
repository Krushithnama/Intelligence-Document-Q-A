from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _model():
    # Lazy import so the backend can still start even if fastembed isn't installed yet.
    from fastembed import TextEmbedding

    # Small, fast default model for local CPU embeddings.
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def embed_texts_local(texts: list[str]) -> np.ndarray:
    """
    Returns float32 matrix of shape (n, d).
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    m = _model()
    vecs = list(m.embed(texts))
    return np.asarray(vecs, dtype=np.float32)

