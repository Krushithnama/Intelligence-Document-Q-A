from __future__ import annotations

import logging

import numpy as np

from backend.embeddings.gemini_client import GeminiClient
from backend.embeddings.local_embedder import embed_texts_local


class Embedder:
    def __init__(self, client: GeminiClient) -> None:
        self._client = client
        self._log = logging.getLogger(__name__)

    def embed(self, texts: list[str]) -> tuple[np.ndarray, str]:
        """
        Returns:
          - vectors: float32 array of shape (n, d)
          - model: embedding model name
        """
        try:
            res = self._client.embed_texts(texts)
            vecs = np.asarray(res.vectors, dtype=np.float32)
            return vecs, res.model
        except Exception as e:
            msg = str(e).lower()
            if "404" not in msg and "not found" not in msg and "not supported for embedcontent" not in msg:
                raise

            # Gemini embeddings unavailable for this key/endpoint → fall back to local embeddings.
            self._log.warning("Gemini embeddings unavailable, falling back to local embeddings: %s", e)
            vecs = embed_texts_local(texts)
            if vecs.ndim != 2 or vecs.shape[0] != len(texts):
                raise RuntimeError("Local embedding backend returned invalid shape") from e
            return vecs.astype(np.float32), "local:BAAI/bge-small-en-v1.5"

