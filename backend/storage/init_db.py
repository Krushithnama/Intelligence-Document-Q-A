from __future__ import annotations

from backend.storage.db import engine
from backend.storage.models import Base


async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

