from __future__ import annotations

from typing import Any, Protocol


class LLMClient(Protocol):
    def chat(self, prompt: str, *, model: str | None = None, temperature: float = 0.2) -> str: ...

    def chat_json(self, prompt: str, *, model: str | None = None, temperature: float = 0.0) -> Any: ...

