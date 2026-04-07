from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import RetryError

from backend.api.deps import (
    answer_cache,
    ingestion_pipeline,
    memory_manager,
    rag_answerer,
    retriever,
)
from backend.api.schemas import (
    AskRequest,
    AskResponse,
    DocumentItem,
    DocumentsListResponse,
    FeedbackRequest,
    FeedbackResponse,
    HistoryResponse,
    UploadResponse,
)
from backend.common.cache import stable_cache_key
from backend.common.config import settings
from backend.common.logging import setup_logging
from backend.feedback.service import FeedbackService
from backend.retrieval.rag import build_context
from backend.storage.db import get_session
from backend.storage.init_db import init_db
from backend.storage.models import ChatMessage, ChatSession, DocChunk, Document


setup_logging()
settings.ensure_dirs()

app = FastAPI(title="Intelligent Document Q&A System", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "intelligent-doc-qa",
        "docs": "/docs",
        "endpoints": ["/upload-document", "/documents", "/ask-question", "/feedback", "/history"],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.on_event("startup")
async def _startup() -> None:
    await init_db()


@app.get("/documents", response_model=DocumentsListResponse)
async def list_documents(
    session: AsyncSession = Depends(get_session),
    limit: int = 100,
):
    lim = max(1, min(limit, 500))
    stmt = (
        select(
            Document.id,
            Document.name,
            Document.content_type,
            Document.created_at,
            func.count(DocChunk.id).label("chunk_count"),
        )
        .outerjoin(DocChunk, DocChunk.document_id == Document.id)
        .group_by(Document.id, Document.name, Document.content_type, Document.created_at)
        .order_by(Document.created_at.desc())
        .limit(lim)
    )
    rows = (await session.execute(stmt)).all()
    items = [
        DocumentItem(
            id=r.id,
            name=r.name,
            content_type=r.content_type,
            created_at=r.created_at.isoformat(),
            chunks=int(r.chunk_count or 0),
        )
        for r in rows
    ]
    return DocumentsListResponse(documents=items)


@app.post("/upload-document", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / file.filename
    content = await file.read()
    target.write_bytes(content)

    pipeline = ingestion_pipeline()
    try:
        res = await pipeline.ingest(
            session=session,
            file_path=target,
            original_filename=file.filename,
            content_type=file.content_type or "application/octet-stream",
        )
        return UploadResponse(document_id=res.document_id, chunks_created=res.chunks_created)
    except Exception as e:
        msg = str(e)
        hint = ""
        lower = msg.lower()
        if "text-embedding-004" in lower or "embedcontent" in lower or "batchembedcontents" in lower:
            hint = (
                " Your embedding model seems unsupported. Set `GEMINI_EMBED_MODEL=embedding-001` in `backend/.env` "
                "(or use a model returned by the API ListModels for embeddings)."
            )
        elif "api key" in lower or "api_key" in lower or "401" in lower or "403" in lower:
            hint = " Check `GEMINI_API_KEY` in `backend/.env` or your environment."
        raise HTTPException(status_code=502, detail=f"Document ingestion failed: {msg}{hint}") from e


@app.post("/ask-question", response_model=AskResponse)
async def ask_question(payload: AskRequest, session: AsyncSession = Depends(get_session)):
    mm = memory_manager()
    cs = await mm.get_or_create_session(session=session, user_id=payload.user_id, session_id=payload.session_id)

    # Cache (optional)
    cache = answer_cache()
    ck = stable_cache_key(
        {
            "user_id": payload.user_id,
            "question": payload.question,
            "doc_ids": payload.doc_ids or [],
            "session_id": cs.id,
        }
    )
    if settings.enable_cache:
        cached = cache.get(ck)
        if cached:
            await mm.add_message(session=session, session_id=cs.id, role="user", content=payload.question)
            await mm.add_message(session=session, session_id=cs.id, role="assistant", content=cached)
            await session.commit()
            return AskResponse(session_id=cs.id, answer=cached, used_chunk_ids=[])

    await mm.add_message(session=session, session_id=cs.id, role="user", content=payload.question)

    try:
        r = retriever()
        chunks = await r.retrieve(session=session, query=payload.question, doc_ids=payload.doc_ids)
        context, used_ids = build_context(chunks)

        st = await mm.short_term_text(session=session, session_id=cs.id)
        lt = await mm.long_term_text(session=session, user_id=payload.user_id, query=payload.question)

        rag = rag_answerer()
        answer = rag.answer(question=payload.question, context=context, short_term=st, long_term=lt)
    except Exception as e:
        # Surface a helpful error instead of a generic 500.
        root: Exception = e
        if isinstance(e, RetryError) and getattr(e, "last_attempt", None) is not None:
            last = e.last_attempt  # type: ignore[assignment]
            exc = last.exception() if hasattr(last, "exception") else None
            if isinstance(exc, Exception):
                root = exc

        hint = ""
        msg = str(root)
        if "API key" in msg or "api_key" in msg or "401" in msg or "403" in msg:
            hint = " Check GEMINI_API_KEY in backend/.env or your environment."
        if "not found" in msg.lower() and "models/" in msg.lower():
            hint = (hint + " " if hint else " ") + "Your GEMINI_CHAT_MODEL may be unavailable for this API key."
        raise HTTPException(status_code=502, detail=f"LLM/RAG pipeline failed: {msg}{hint}".strip()) from root

    await mm.add_message(session=session, session_id=cs.id, role="assistant", content=answer)

    # Memory extraction happens after answering.
    if os.environ.get("DISABLE_LONG_TERM_EXTRACTION") != "1":
        await mm.extract_and_store_long_term(
            session=session,
            user_id=payload.user_id,
            user_text=payload.question,
            assistant_text=answer,
        )

    # Periodic episodic summary update (simple heuristic).
    if os.environ.get("DISABLE_SESSION_SUMMARY") != "1":
        await mm.update_session_summary(session=session, session_id=cs.id)

    await session.commit()

    if settings.enable_cache:
        cache.set(ck, answer)

    return AskResponse(session_id=cs.id, answer=answer, used_chunk_ids=used_ids)


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(payload: FeedbackRequest, session: AsyncSession = Depends(get_session)):
    if payload.rating not in (-1, 1):
        raise HTTPException(status_code=400, detail="rating must be +1 or -1")
    svc = FeedbackService()
    fid = await svc.record(
        session=session,
        user_id=payload.user_id,
        session_id=payload.session_id,
        question=payload.question,
        answer=payload.answer,
        rating=payload.rating,
        correction=payload.correction,
        doc_ids=payload.doc_ids,
        chunk_ids=payload.chunk_ids,
    )
    return FeedbackResponse(feedback_id=fid)


@app.get("/history", response_model=HistoryResponse)
async def history(user_id: str = "default", session_id: str | None = None, session: AsyncSession = Depends(get_session)):
    mm = memory_manager()
    cs = await mm.get_or_create_session(session=session, user_id=user_id, session_id=session_id)
    msgs = (
        await session.execute(
            select(ChatMessage).where(ChatMessage.session_id == cs.id).order_by(ChatMessage.created_at.asc())
        )
    ).scalars().all()
    sess = (await session.execute(select(ChatSession).where(ChatSession.id == cs.id))).scalar_one()
    return HistoryResponse(
        session_id=cs.id,
        messages=[{"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()} for m in msgs],
        summary=sess.summary,
    )

