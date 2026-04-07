from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.config import settings
from backend.embeddings.embedder import Embedder
from backend.retrieval.vector_store import FaissDocIndex, VectorHit
from backend.storage.models import DocChunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    doc_name: str | None
    section: str | None
    text: str
    score: float


class Retriever:
    def __init__(self, *, embedder: Embedder, doc_index: FaissDocIndex) -> None:
        self._embedder = embedder
        self._doc_index = doc_index

    async def retrieve(
        self,
        *,
        session: AsyncSession,
        query: str,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
        rerank_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Two-stage retrieval:
        1) Vector search over FAISS (broad)
        2) DB fetch + metadata filtering + score-based rerank
        """
        tk = top_k or settings.top_k
        rk = rerank_k or settings.rerank_k
        broad_k = max(rk, tk) * 5

        qv, _ = self._embedder.embed([query])
        hits: list[VectorHit] = self._doc_index.search(query_vector=qv[0], top_k=broad_k)
        if not hits:
            return []

        # Filter by doc_ids using stored metadata when present.
        if doc_ids:
            doc_id_set = set(doc_ids)
            hits = [h for h in hits if h.metadata.get("document_id") in doc_id_set]
            if not hits:
                return []

        # Fetch chunk texts from DB for the remaining hit ids.
        hit_ids = [h.id for h in hits[:rk]]
        rows = (
            await session.execute(select(DocChunk).where(DocChunk.id.in_(hit_ids)))  # noqa: E711
        ).scalars().all()
        row_by_id = {r.id: r for r in rows}

        out: list[RetrievedChunk] = []
        for h in hits[:rk]:
            r = row_by_id.get(h.id)
            if not r:
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=r.id,
                    document_id=r.document_id,
                    doc_name=h.metadata.get("doc_name"),
                    section=r.section,
                    text=r.text,
                    score=h.score,
                )
            )

        # Score-based rerank baseline (stable + cheap).
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:tk]

