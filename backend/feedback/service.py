from __future__ import annotations

import datetime as dt

from sqlalchemy.ext.asyncio import AsyncSession

from backend.storage.models import Feedback


class FeedbackService:
    async def record(
        self,
        *,
        session: AsyncSession,
        user_id: str,
        session_id: str | None,
        question: str,
        answer: str,
        rating: int,
        correction: str | None,
        doc_ids: list[str] | None,
        chunk_ids: list[str] | None,
    ) -> str:
        r = Feedback(
            user_id=user_id,
            session_id=session_id,
            question=question,
            answer=answer,
            rating=1 if rating >= 1 else -1,
            correction=correction,
            doc_ids=doc_ids or [],
            chunk_ids=chunk_ids or [],
            created_at=dt.datetime.now(dt.UTC),
        )
        session.add(r)
        await session.flush()
        await session.commit()
        return r.id

