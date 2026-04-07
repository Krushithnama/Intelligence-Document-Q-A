# Intelligent Document Q&A System with Memory (Gemini + RAG)

Production-ready, modular RAG system that:
- Ingests PDF/DOCX/TXT documents
- Builds a vector index (FAISS) using **Gemini embeddings**
- Answers questions with **Retrieval-Augmented Generation**
- Maintains **short-term, long-term, and episodic memory**
- Learns from **feedback** (upvote/downvote + corrections)
- Includes caching + evaluation scripts

## Repo layout

- `backend/`: FastAPI service (ingestion, embeddings, retrieval, memory, API)
- `frontend/`: Streamlit UI (basic)
- `scripts/`: evaluation + utilities
- `data/`: local persistence (SQLite DB, FAISS indexes, uploads)

## Prerequisites

- Python 3.11+
- A Google Gemini API key

Set an environment variable:

- PowerShell:
  - `setx GEMINI_API_KEY "YOUR_KEY"`
  - restart your terminal after running `setx`

## Quickstart

### 1) Create and activate a venv

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the API

```bash
uvicorn backend.api.main:app --reload --port 8000
```

Swagger UI:
- `http://localhost:8000/docs`

### 4) Run the UI

```bash
streamlit run frontend\app.py
```

## One-command run (PowerShell)

- Backend: `.\run_backend.ps1`
- Frontend (after backend is running): `.\run_frontend.ps1`

## Core API endpoints

- `POST /upload-document`
- `POST /ask-question`
- `POST /feedback`
- `GET  /history`

## Example flow

1) Upload a PDF/DOCX/TXT via UI or `POST /upload-document`
2) Ask:
   - “What are the key requirements and deadlines?”
   - “Summarize section 3 and list risks.”
3) Upvote/downvote, optionally provide a correction
4) Ask follow-ups; the system uses short/long-term memory to stay consistent

## Notes

- Default storage is **SQLite** at `data/app.db`. You can switch to Postgres via env vars in `backend/.env.example`.
- Vector indexes are stored under `data/indexes/`.

