from __future__ import annotations

from backend.ingestion.chunking import chunk_text


def test_chunk_text_produces_chunks() -> None:
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = chunk_text(text, target_chars=20, overlap_chars=5, max_chars=30)
    assert len(chunks) >= 2
    assert all(c.text.strip() for c in chunks)

