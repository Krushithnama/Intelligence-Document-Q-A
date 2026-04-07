from __future__ import annotations

from functools import lru_cache

from backend.common.cache import AnswerCache
from backend.common.config import settings
from backend.embeddings.embedder import Embedder
from backend.embeddings.gemini_client import GeminiClient
from backend.ingestion.pipeline import DocumentIngestionPipeline
from backend.llm.base import LLMClient
from backend.llm.ollama_client import OllamaClient
from backend.memory.manager import MemoryManager
from backend.retrieval.rag import RagAnswerer
from backend.retrieval.retriever import Retriever
from backend.retrieval.vector_store import FaissDocIndex, FaissMemoryIndex


@lru_cache(maxsize=1)
def gemini_client() -> GeminiClient:
    return GeminiClient()

@lru_cache(maxsize=1)
def llm_client() -> LLMClient:
    if settings.llm_provider == "ollama":
        return OllamaClient()
    return gemini_client()


@lru_cache(maxsize=1)
def embedder() -> Embedder:
    return Embedder(gemini_client())


@lru_cache(maxsize=1)
def doc_index() -> FaissDocIndex:
    return FaissDocIndex(index_dir=settings.doc_index_dir)


@lru_cache(maxsize=1)
def memory_index() -> FaissMemoryIndex:
    return FaissMemoryIndex(index_dir=settings.memory_index_dir)


@lru_cache(maxsize=1)
def retriever() -> Retriever:
    return Retriever(embedder=embedder(), doc_index=doc_index())


@lru_cache(maxsize=1)
def rag_answerer() -> RagAnswerer:
    return RagAnswerer(llm=llm_client())


@lru_cache(maxsize=1)
def ingestion_pipeline() -> DocumentIngestionPipeline:
    return DocumentIngestionPipeline(embedder=embedder(), doc_index=doc_index())


@lru_cache(maxsize=1)
def memory_manager() -> MemoryManager:
    return MemoryManager(llm=llm_client(), embedder=embedder(), memory_index=memory_index())


@lru_cache(maxsize=1)
def answer_cache() -> AnswerCache:
    return AnswerCache.create()

