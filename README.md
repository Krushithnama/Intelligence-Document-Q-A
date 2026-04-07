# 🧠 Intelligent Document Q&A System

A **production-ready, modular Document Q&A system** that combines **Retrieval-Augmented Generation (RAG)** with a **memory layer and feedback loop** to deliver context-aware, evolving answers.

This project is designed to reflect **real-world system design principles**:

* Clean architecture
* Modular components
* Scalable pipeline
* Practical AI integration (local + cloud fallback)

---

# 🚀 Core Capabilities

### 📄 Document Understanding

* Ingests **PDF / DOCX / TXT**
* Intelligent chunking (semantic + fixed size)
* Metadata tracking (document name, section, timestamps)

---

### 🔍 Retrieval-Augmented Generation (RAG)

Pipeline:

```
User Query → Embedding → Vector Search → Re-ranking → LLM → Answer
```

* FAISS-based vector search
* Top-K retrieval + re-ranking
* Context-aware answer generation

---

### 🧠 Memory System (Key Requirement)

Implements **multi-layer memory**:

#### Short-Term Memory

* Stores last N conversation turns

#### Long-Term Memory

* Stores important facts, preferences, past queries
* Retrieved using semantic similarity

#### Episodic Memory

* Stores full conversation sessions + outcomes

---

### 🔁 Feedback Learning Loop

* Upvote / Downvote responses
* User corrections stored
* Improves future retrieval + ranking

---

### ⚡ Performance Optimization

* Query caching
* Precomputed embeddings
* Efficient context window control

---

# 🏗️ Tech Stack

### Backend

* FastAPI
* Python 3.11+

### LLM Options (Flexible)

#### ✅ Option 1: Local 

* Ollama (no API cost)
* Example model:

  ```
  llama3.2:latest
  ```

#### ✅ Option 2: Cloud (Optional)

* Gemini API (for embeddings & generation)

---

### Vector Storage

* FAISS 

---

### Database

* SQLite (default)
* Optional: PostgreSQL

---

### Frontend

* Streamlit (ChatGPT-style UI)

---

# 📁 Project Structure

```
backend/
  ├── api/            # FastAPI routes
  ├── ingestion/      # Document processing
  ├── embeddings/     # Embedding logic (Gemini + fallback)
  ├── retrieval/      # RAG pipeline
  ├── memory/         # Memory system
  └── core/           # Config, utils

frontend/
  └── app.py          # Streamlit UI

scripts/
  └── evaluation/     # Metrics & testing

data/ (gitignored)
  ├── app.db
  ├── indexes/
  └── uploads/
```

---

# ⚙️ Setup Guide

## 1. Create Virtual Environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Configure Environment

Copy:

```
backend/.env.example → backend/.env
```

---

## 🔑 Configuration Logic (IMPORTANT)

### Scenario A: Using Gemini API 

If you have a Gemini API key:

```
GEMINI_API_KEY=your_key_here
LLM_PROVIDER=gemini
```

* Enables cloud embeddings + generation
* Better semantic understanding

---

### Scenario B: No API Key 

If you **do NOT have Gemini API key**:

```
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:latest
```

👉 REQUIREMENT:

* Ollama must be installed and running locally

Check:

```bash
ollama list
```

---

### ⚡ Hybrid Behavior (Smart Fallback)

* If Gemini embeddings fail → system falls back to local embeddings
* Ensures **robustness and no hard dependency on cloud**

---

# ▶️ Running the System

## Start Backend

```bash
uvicorn backend.api.main:app --reload --port 8000
```

Docs:

```
http://127.0.0.1:8000/docs
```

---

## Start Frontend

```bash
streamlit run frontend/app.py
```

---

## One-Command Scripts (Windows)

```bash
.\run_backend.ps1
.\run_frontend.ps1
```

---

# 🔌 API Endpoints

| Endpoint         | Method | Description      |
| ---------------- | ------ | ---------------- |
| /upload-document | POST   | Upload files     |
| /documents       | GET    | List documents   |
| /ask-question    | POST   | Ask questions    |
| /feedback        | POST   | Provide feedback |
| /history         | GET    | Chat history     |

---

# 🔄 Example Workflow

1. Upload document
2. Ask:

   * “Summarize section 3”
   * “List key risks and deadlines”
3. System:

   * Retrieves relevant chunks
   * Uses memory + context
   * Generates answer
4. Provide feedback
5. Ask follow-ups → system improves

---

# 📊 Evaluation System

Includes scripts for:

* Semantic similarity scoring
* Retrieval accuracy
* Response latency

---

# ⚠️ Important Notes

### Security

* `.env` is gitignored
* Never commit API keys

---

### Storage

* SQLite → `data/app.db`
* FAISS → `data/indexes/`

---

### Model Selection

| Component       | Purpose           |
| --------------- | ----------------- |
| Chat Model      | Answer generation |
| Embedding Model | Retrieval         |

---

# 🧠 System Design Insight

This project demonstrates:

* RAG architecture (industry standard)
* Memory-augmented systems
* Feedback-driven improvement
* Modular backend design

---

# 🚀 Future Enhancements

* Multi-user support
* Advanced reranking (cross-encoder)
* Streaming responses
* Deployment (Docker + Cloud)

---

# ✅ Summary

This system is:

* ✔️ Production-oriented
* ✔️ Flexible (Local + Cloud)
* ✔️ Robust (fallback mechanisms)
* ✔️ Interview-ready (system design depth)

---

👉 If you have Gemini API → use it for better performance
👉 If not → Ollama ensures full local execution

Both paths are fully supported by design.
