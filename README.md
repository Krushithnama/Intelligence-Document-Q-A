# Intelligent Document Q&A

Production-ready, modular document Q&A app that:
- Ingests PDF/DOCX/TXT documents
- Builds a vector index (FAISS)
- Answers questions with **Retrieval-Augmented Generation (RAG)**
- Uses **local Ollama** for answering (no cloud quota required)
- Falls back to **local embeddings** automatically if cloud embeddings are unavailable
- Includes optional caching + evaluation scripts

## Repo layout

- `backend/`: FastAPI service (ingestion, embeddings, retrieval, memory, API)
- `frontend/`: Streamlit UI (ChatGPT-like)
- `scripts/`: evaluation + utilities
- `data/`: local persistence (SQLite DB, FAISS indexes, uploads) (gitignored)

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running (recommended)

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

### 3) Configure backend

Copy `backend/.env.example` to `backend/.env` and set:

- `GEMINI_API_KEY` (optional; embeddings may be unavailable for some keys)
- `LLM_PROVIDER=ollama`
- `OLLAMA_MODEL=llama3.2:latest` (or any model you have in `ollama list`)

### 4) Run the API

```bash
uvicorn backend.api.main:app --reload --port 8000
```

Swagger UI:
- `http://127.0.0.1:8000/docs`

### 5) Run the UI

```bash
streamlit run frontend\app.py
```

## One-command run (PowerShell)

- Backend: `.\run_backend.ps1`
- Frontend (after backend is running): `.\run_frontend.ps1`

## Core API endpoints

- `POST /upload-document`
- `GET  /documents`
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
- Do **not** commit secrets: `backend/.env` is gitignored.

