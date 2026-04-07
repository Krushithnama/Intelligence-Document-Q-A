from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from cachetools import TTLCache

from backend.common.config import settings


def stable_cache_key(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class AnswerCache:
    """
    In-process TTL cache (optional). You can swap this with Redis later.
    """

    _cache: TTLCache

    @classmethod
    def create(cls) -> "AnswerCache":
        ttl = settings.cache_ttl_seconds
        return cls(_cache=TTLCache(maxsize=2048, ttl=ttl))

    def get(self, key: str) -> str | None:
        return self._cache.get(key)

    def set(self, key: str, value: str) -> None:
        self._cache[key] = value

