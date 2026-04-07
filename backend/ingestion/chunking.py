from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    text: str
    section: str | None
    char_start: int | None
    char_end: int | None


def _sentences(text: str) -> list[str]:
    """
    Lightweight sentence splitter (no external NLP deps).
    It’s intentionally conservative and falls back to line breaks.
    """
    import re

    text = text.strip()
    if not text:
        return []
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Split on sentence-ish boundaries.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    if len(parts) <= 1:
        parts = re.split(r"\n{2,}", text)
    return [p.strip() for p in parts if p and p.strip()]


def chunk_text(
    text: str,
    *,
    section: str | None = None,
    target_chars: int = 1200,
    overlap_chars: int = 150,
    max_chars: int = 1800,
) -> list[TextChunk]:
    """
    Hybrid chunking:
    - Build semantically coherent chunks by grouping sentence units
    - Enforce upper bound with a hard split when needed
    - Add a small overlap to preserve context across boundaries
    """
    text = text.strip()
    if not text:
        return []

    units = _sentences(text)
    chunks: list[str] = []
    buf = ""
    for u in units:
        if not buf:
            buf = u
            continue
        if len(buf) + 1 + len(u) <= target_chars:
            buf = f"{buf} {u}"
        else:
            chunks.append(buf)
            buf = u
    if buf:
        chunks.append(buf)

    # Hard split oversize chunks.
    final_chunks: list[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            final_chunks.append(c)
        else:
            start = 0
            while start < len(c):
                end = min(start + max_chars, len(c))
                final_chunks.append(c[start:end].strip())
                start = max(0, end - overlap_chars)

    # Add overlap between adjacent chunks (soft overlap on boundaries).
    overlapped: list[str] = []
    for i, c in enumerate(final_chunks):
        if i == 0 or overlap_chars <= 0:
            overlapped.append(c)
            continue
        prev = overlapped[-1]
        ov = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
        merged = (ov + "\n" + c).strip()
        overlapped.append(merged)

    # char_start/char_end are best-effort for section text, not exact for overlapped chunks.
    out: list[TextChunk] = []
    cursor = 0
    for c in overlapped:
        idx = text.find(c[: min(80, len(c))], cursor)  # fuzzy anchor
        if idx == -1:
            idx = None
        if idx is None:
            out.append(TextChunk(text=c, section=section, char_start=None, char_end=None))
        else:
            out.append(TextChunk(text=c, section=section, char_start=idx, char_end=min(idx + len(c), len(text))))
            cursor = idx + max(1, len(c) // 4)
    return out

