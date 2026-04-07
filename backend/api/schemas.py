from __future__ import annotations

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    chunks_created: int


class AskRequest(BaseModel):
    user_id: str = Field(default="default")
    session_id: str | None = None
    question: str
    doc_ids: list[str] | None = None


class AskResponse(BaseModel):
    session_id: str
    answer: str
    used_chunk_ids: list[str]


class FeedbackRequest(BaseModel):
    user_id: str = Field(default="default")
    session_id: str | None = None
    question: str
    answer: str
    rating: int = Field(description="+1 for upvote, -1 for downvote")
    correction: str | None = None
    doc_ids: list[str] | None = None
    chunk_ids: list[str] | None = None


class FeedbackResponse(BaseModel):
    feedback_id: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[dict]
    summary: str | None


class DocumentItem(BaseModel):
    id: str
    name: str
    content_type: str
    created_at: str
    chunks: int


class DocumentsListResponse(BaseModel):
    documents: list[DocumentItem]

