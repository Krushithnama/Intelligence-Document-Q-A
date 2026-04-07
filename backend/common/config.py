from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the backend service.

    Values come from environment variables and/or an optional `.env`.
    """

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str
    gemini_chat_model: str = "gemini-2.0-flash"
    # Preferred embedding model. Availability can vary by key/endpoint; the client will try fallbacks.
    gemini_embed_model: str = "gemini-embedding-001"

    llm_provider: Literal["gemini", "ollama"] = "gemini"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1"
    ollama_timeout_seconds: int = 120

    database_url: str = "sqlite+aiosqlite:///../data/app.db"

    doc_index_dir: str = "../data/indexes/docs"
    memory_index_dir: str = "../data/indexes/memory"
    upload_dir: str = "../data/uploads"

    top_k: int = 8
    rerank_k: int = 16
    max_context_chars: int = 12000

    short_term_turns: int = 10
    long_term_top_k: int = 6

    enable_cache: bool = True
    cache_ttl_seconds: int = 900
    cache_backend: Literal["memory"] = "memory"

    def ensure_dirs(self) -> None:
        for p in (self.doc_index_dir, self.memory_index_dir, self.upload_dir):
            Path(p).mkdir(parents=True, exist_ok=True)


settings = Settings()

