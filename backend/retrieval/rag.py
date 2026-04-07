from __future__ import annotations

from dataclasses import dataclass

from backend.common.config import settings
from backend.llm.base import LLMClient
from backend.retrieval.retriever import RetrievedChunk


@dataclass(frozen=True)
class AnswerWithSources:
    answer: str
    used_chunk_ids: list[str]


def build_context(chunks: list[RetrievedChunk], *, max_chars: int | None = None) -> tuple[str, list[str]]:
    """
    Builds a compact context block with stable chunk ids for citation.
    """
    max_c = max_chars or settings.max_context_chars
    parts: list[str] = []
    used: list[str] = []
    total = 0
    for c in chunks:
        header = f"[chunk_id={c.chunk_id} doc={c.doc_name or c.document_id} section={c.section or 'N/A'} score={c.score:.3f}]"
        block = header + "\n" + c.text.strip()
        if total + len(block) + 2 > max_c:
            break
        parts.append(block)
        used.append(c.chunk_id)
        total += len(block) + 2
    return "\n\n---\n\n".join(parts), used


class RagAnswerer:
    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    def answer(
        self,
        *,
        question: str,
        context: str,
        short_term: str,
        long_term: str,
    ) -> str:
        prompt = f"""
You are a production-grade assistant for document Q&A using Retrieval-Augmented Generation.

Rules:
- Answer ONLY using the provided CONTEXT when possible. If context is insufficient, say what is missing.
- Be precise, avoid hallucination.
- When you use facts from context, cite chunk ids like: (chunk_id=...)
- Keep the answer concise but complete.

SHORT-TERM MEMORY (recent conversation):
{short_term.strip() or "[none]"}

LONG-TERM MEMORY (user facts/preferences):
{long_term.strip() or "[none]"}

CONTEXT:
{context.strip() or "[no context retrieved]"}

QUESTION:
{question.strip()}
""".strip()
        return self._llm.chat(prompt, temperature=0.2)

