from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return (v / norms).astype(np.float32)


@dataclass(frozen=True)
class VectorHit:
    id: str
    score: float
    metadata: dict[str, Any]


class _FaissIndexBase:
    def __init__(self, *, index_dir: str) -> None:
        self._dir = Path(index_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "index.faiss"
        self._meta_path = self._dir / "meta.jsonl"
        self._dim_path = self._dir / "dim.json"

        self._index: faiss.Index | None = None
        self._id_to_pos: dict[str, int] = {}
        self._pos_to_id: list[str] = []
        self._pos_to_meta: list[dict[str, Any]] = []
        self._dim: int | None = None

        self._load()

    @property
    def dim(self) -> int | None:
        return self._dim

    def _make_index(self, dim: int) -> faiss.Index:
        # Cosine similarity via inner product on normalized vectors.
        return faiss.IndexFlatIP(dim)

    def _load(self) -> None:
        if self._dim_path.exists():
            self._dim = json.loads(self._dim_path.read_text(encoding="utf-8"))["dim"]
        if self._meta_path.exists():
            self._pos_to_id = []
            self._pos_to_meta = []
            self._id_to_pos = {}
            for line in self._meta_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                _id = row["id"]
                meta = row.get("metadata", {})
                pos = len(self._pos_to_id)
                self._pos_to_id.append(_id)
                self._pos_to_meta.append(meta)
                self._id_to_pos[_id] = pos
        if self._index_path.exists():
            self._index = faiss.read_index(str(self._index_path))
        elif self._dim is not None:
            self._index = self._make_index(self._dim)

    def persist(self) -> None:
        if self._index is None or self._dim is None:
            return
        faiss.write_index(self._index, str(self._index_path))
        self._dim_path.write_text(json.dumps({"dim": self._dim}), encoding="utf-8")
        with self._meta_path.open("w", encoding="utf-8") as f:
            for _id, meta in zip(self._pos_to_id, self._pos_to_meta, strict=True):
                f.write(json.dumps({"id": _id, "metadata": meta}, ensure_ascii=False) + "\n")

    def add(self, *, vectors: np.ndarray, ids: list[str], metadatas: list[dict[str, Any]]) -> None:
        if len(ids) != len(metadatas):
            raise ValueError("ids and metadatas length mismatch")
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D")
        if vectors.shape[0] != len(ids):
            raise ValueError("vectors and ids length mismatch")

        vectors = vectors.astype(np.float32)
        vectors = _l2_normalize(vectors)
        dim = int(vectors.shape[1])
        if self._dim is None:
            self._dim = dim
            self._index = self._make_index(dim)
        if dim != self._dim:
            raise ValueError(f"Vector dim mismatch: got {dim} expected {self._dim}")
        assert self._index is not None

        for _id in ids:
            if _id in self._id_to_pos:
                # For simplicity: skip duplicates (id stability is important).
                continue

        kept_vecs = []
        kept_ids = []
        kept_metas = []
        for i, _id in enumerate(ids):
            if _id in self._id_to_pos:
                continue
            kept_vecs.append(vectors[i])
            kept_ids.append(_id)
            kept_metas.append(metadatas[i])

        if not kept_ids:
            return
        mat = np.vstack(kept_vecs).astype(np.float32)
        self._index.add(mat)
        for _id, meta in zip(kept_ids, kept_metas, strict=True):
            pos = len(self._pos_to_id)
            self._pos_to_id.append(_id)
            self._pos_to_meta.append(meta)
            self._id_to_pos[_id] = pos

    def search(self, *, query_vector: np.ndarray, top_k: int) -> list[VectorHit]:
        if self._index is None or self._dim is None or self._index.ntotal == 0:
            return []
        q = query_vector.astype(np.float32).reshape(1, -1)
        if q.shape[1] != self._dim:
            raise ValueError(f"Query dim mismatch: got {q.shape[1]} expected {self._dim}")
        q = _l2_normalize(q)
        scores, idxs = self._index.search(q, top_k)
        hits: list[VectorHit] = []
        for score, pos in zip(scores[0].tolist(), idxs[0].tolist(), strict=True):
            if pos < 0:
                continue
            _id = self._pos_to_id[pos]
            meta = self._pos_to_meta[pos]
            hits.append(VectorHit(id=_id, score=float(score), metadata=meta))
        return hits


class FaissDocIndex(_FaissIndexBase):
    pass


class FaissMemoryIndex(_FaissIndexBase):
    pass

