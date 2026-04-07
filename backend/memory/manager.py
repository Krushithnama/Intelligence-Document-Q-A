from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.config import settings
from backend.embeddings.embedder import Embedder
from backend.llm.base import LLMClient
from backend.memory.prompts import MEMORY_EXTRACTION_PROMPT
from backend.retrieval.vector_store import FaissMemoryIndex
from backend.storage.models import ChatMessage, ChatSession, MemoryItem


class MemoryManager:
    """
    - Short-term memory: last N messages from the session
    - Long-term memory: durable MemoryItem records, retrieved semantically via FAISS
    - Episodic memory: ChatSession summary/outcome stored in DB
    """

    def __init__(self, *, llm: LLMClient, embedder: Embedder, memory_index: FaissMemoryIndex) -> None:
        self._llm = llm
        self._embedder = embedder
        self._memory_index = memory_index

    async def get_or_create_session(self, *, session: AsyncSession, user_id: str, session_id: str | None) -> ChatSession:
        if session_id:
            existing = (await session.execute(select(ChatSession).where(ChatSession.id == session_id))).scalar_one_or_none()
            if existing:
                return existing
        cs = ChatSession(user_id=user_id, title=None)
        session.add(cs)
        await session.flush()
        return cs

    async def add_message(self, *, session: AsyncSession, session_id: str, role: str, content: str) -> None:
        session.add(ChatMessage(session_id=session_id, role=role, content=content))
        await session.flush()

    async def short_term_text(self, *, session: AsyncSession, session_id: str) -> str:
        n = settings.short_term_turns * 2
        msgs = (
            await session.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.desc())
                .limit(n)
            )
        ).scalars().all()
        msgs = list(reversed(msgs))
        return "\n".join([f"{m.role}: {m.content}" for m in msgs])

    async def long_term_text(self, *, session: AsyncSession, user_id: str, query: str) -> str:
        if self._memory_index.dim is None or self._memory_index.dim == 0:
            return ""
        qv, _ = self._embedder.embed([query])
        hits = self._memory_index.search(query_vector=qv[0], top_k=settings.long_term_top_k * 4)
        if not hits:
            return ""

        ids = [h.id for h in hits]
        rows = (await session.execute(select(MemoryItem).where(MemoryItem.id.in_(ids), MemoryItem.user_id == user_id))).scalars().all()  # noqa: E711
        by_id = {r.id: r for r in rows}

        picked = []
        for h in hits:
            r = by_id.get(h.id)
            if not r:
                continue
            picked.append((h.score, r))
        picked.sort(key=lambda t: (t[0] * 0.7 + t[1].importance * 0.3), reverse=True)
        picked = picked[: settings.long_term_top_k]

        now = dt.datetime.now(dt.UTC)
        for _, r in picked:
            r.last_used_at = now
        await session.flush()

        return "\n".join([f"- ({r.kind}, importance={r.importance:.2f}) {r.content}" for _, r in picked])

    async def extract_and_store_long_term(
        self,
        *,
        session: AsyncSession,
        user_id: str,
        user_text: str,
        assistant_text: str,
    ) -> int:
        prompt = (
            MEMORY_EXTRACTION_PROMPT
            + "\n\nCONVERSATION TURN:\n"
            + f"user: {user_text}\nassistant: {assistant_text}\n"
        )
        try:
            data = self._llm.chat_json(prompt, temperature=0.0)
        except Exception:
            return 0

        items = data.get("items", []) if isinstance(data, dict) else []
        to_store = []
        for it in items:
            try:
                kind = str(it.get("kind", "other"))
                content = str(it.get("content", "")).strip()
                importance = float(it.get("importance", 0.5))
                meta = it.get("metadata") or {}
            except Exception:
                continue
            if not content:
                continue
            if importance < 0.6:
                continue
            to_store.append((kind, content, max(0.0, min(1.0, importance)), meta))

        if not to_store:
            return 0

        texts = [c for _, c, _, _ in to_store]
        vecs, model = self._embedder.embed(texts)
        dim = int(vecs.shape[1])

        rows: list[MemoryItem] = []
        for (kind, content, importance, meta) in to_store:
            rows.append(
                MemoryItem(
                    user_id=user_id,
                    kind=kind,
                    content=content,
                    importance=importance,
                    embedding_model=model,
                    embedding_dim=dim,
                    source="extraction",
                    meta=meta,
                )
            )
            session.add(rows[-1])
        await session.flush()

        ids = [r.id for r in rows]
        metas = [{"user_id": user_id, "kind": r.kind} for r in rows]
        self._memory_index.add(vectors=vecs, ids=ids, metadatas=metas)
        self._memory_index.persist()
        await session.flush()
        return len(rows)

    async def update_session_summary(self, *, session: AsyncSession, session_id: str) -> None:
        """
        Lightweight episodic memory: keep a running summary for the session.
        """
        msgs = (
            await session.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
                .limit(40)
            )
        ).scalars().all()
        text = "\n".join([f"{m.role}: {m.content}" for m in msgs])
        prompt = f"""
Summarize this conversation in 5-10 bullet points.
Return plain text bullets.

Conversation:
{text}
""".strip()
        summary = self._llm.chat(prompt, temperature=0.2)
        cs = (await session.execute(select(ChatSession).where(ChatSession.id == session_id))).scalar_one()
        cs.summary = summary
        await session.flush()

