from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from backend.common.config import settings
from backend.embeddings.embedder import Embedder
from backend.ingestion.chunking import chunk_text
from backend.ingestion.parsers import parse_file
from backend.retrieval.vector_store import FaissDocIndex
from backend.storage.models import DocChunk, Document


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    chunks_created: int


class DocumentIngestionPipeline:
    def __init__(self, *, embedder: Embedder, doc_index: FaissDocIndex) -> None:
        self._embedder = embedder
        self._doc_index = doc_index

    async def ingest(
        self,
        *,
        session: AsyncSession,
        file_path: Path,
        original_filename: str,
        content_type: str,
    ) -> IngestResult:
        parsed = parse_file(file_path, content_type=content_type)

        doc = Document(
            name=original_filename,
            content_type=content_type,
            source_path=str(file_path),
        )
        session.add(doc)
        await session.flush()  # assign doc.id

        chunk_texts: list[str] = []
        chunk_rows: list[DocChunk] = []
        chunk_idx = 0
        for section_title, section_text in parsed.sections:
            for ch in chunk_text(section_text, section=section_title):
                if not ch.text.strip():
                    continue
                chunk_texts.append(ch.text)
                chunk_rows.append(
                    DocChunk(
                        document_id=doc.id,
                        chunk_index=chunk_idx,
                        section=ch.section,
                        text=ch.text,
                        char_start=ch.char_start,
                        char_end=ch.char_end,
                        embedding_model=settings.gemini_embed_model,
                        embedding_dim=0,  # set after embedding
                        meta={"original_filename": original_filename},
                    )
                )
                chunk_idx += 1

        if not chunk_rows:
            await session.commit()
            return IngestResult(document_id=doc.id, chunks_created=0)

        vecs, model = self._embedder.embed(chunk_texts)
        dim = int(vecs.shape[1])
        for r in chunk_rows:
            r.embedding_model = model
            r.embedding_dim = dim
            session.add(r)
        await session.flush()  # assign chunk ids

        ids = [r.id for r in chunk_rows]
        metas = [
            {
                "document_id": r.document_id,
                "chunk_id": r.id,
                "doc_name": original_filename,
                "section": r.section,
            }
            for r in chunk_rows
        ]
        self._doc_index.add(vectors=vecs, ids=ids, metadatas=metas)
        self._doc_index.persist()

        await session.commit()
        return IngestResult(document_id=doc.id, chunks_created=len(chunk_rows))

