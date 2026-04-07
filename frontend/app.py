from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests
import streamlit as st


APP_TITLE = "Intelligent Document Q&A"


@dataclass(frozen=True)
class UploadedDoc:
    document_id: str
    name: str
    uploaded_at: str


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _api_base() -> str:
    return st.session_state.get("api_base", "http://127.0.0.1:8000").rstrip("/")


def _user_id() -> str:
    return st.session_state.get("user_id", "default").strip() or "default"


def _http(method: str, path: str, *, json: Any | None = None, files: Any | None = None, params: Any | None = None, timeout: int = 120):
    url = f"{_api_base()}{path}"
    return requests.request(method, url, json=json, files=files, params=params, timeout=timeout)


def _init_state() -> None:
    st.session_state.setdefault("api_base", "http://127.0.0.1:8000")
    st.session_state.setdefault("user_id", "default")
    st.session_state.setdefault("session_id", "")
    st.session_state.setdefault("uploaded_docs", [])
    st.session_state.setdefault("active_doc_ids", [])
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("last_upload", None)


_init_state()

st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="wide")

st.markdown(
    """
<style>
  /* Give room so the centered header is never clipped */
  .block-container { padding-top: 2.6rem; padding-bottom: 2.5rem; max-width: 1040px; }
  .small-muted { color: rgba(49, 51, 63, 0.65); font-size: 0.9rem; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
  .pill {
    display: inline-block;
    padding: 0.22rem 0.55rem;
    border-radius: 999px;
    background: rgba(14, 17, 23, 0.06);
    font-size: 0.85rem;
  }
  .card {
    border: 1px solid rgba(49, 51, 63, 0.12);
    border-radius: 14px;
    padding: 1rem 1rem;
    background: rgba(255, 255, 255, 0.6);
  }
  .section-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 0.2rem; }
  .stChatMessage { border-radius: 14px; }

  /* ChatGPT-like alignment: user right, assistant left */
  div[data-testid="stChatMessage"] { width: 100%; }
  div[data-testid="stChatMessage"][aria-label="user"] { justify-content: flex-end; }
  div[data-testid="stChatMessage"][aria-label="assistant"] { justify-content: flex-start; }

  /* Make user bubble sit to the right (best-effort across Streamlit versions) */
  div[data-testid="stChatMessage"][aria-label="user"] > div { flex-direction: row-reverse; }
  div[data-testid="stChatMessage"][aria-label="user"] [data-testid="stChatMessageContent"] {
    background: rgba(14, 17, 23, 0.06);
    border-radius: 14px;
    padding: 0.6rem 0.8rem;
  }
  div[data-testid="stChatMessage"][aria-label="assistant"] [data-testid="stChatMessageContent"] {
    border-radius: 14px;
    padding: 0.6rem 0.8rem;
  }

  /* Hide Streamlit chrome (Deploy button / toolbars / footer) */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header { visibility: hidden; }
  [data-testid="stToolbar"] { display: none; }
  [data-testid="stDeployButton"] { display: none; }
  [data-testid="stStatusWidget"] { display: none; }

  /* Reduce extra "bar" feel around uploader */
  [data-testid="stFileUploader"] section { padding: 0 !important; }
  [data-testid="stFileUploaderDropzone"] { border-radius: 14px; }

</style>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown(f"## {APP_TITLE}")
    with st.expander("Settings", expanded=False):
        st.session_state["api_base"] = st.text_input("API Base URL", value=st.session_state["api_base"])
        st.session_state["user_id"] = st.text_input("User ID", value=st.session_state["user_id"])
    st.caption("Upload a document, then chat below.")

st.markdown(
    f"""
<div style="text-align:center; margin-top: 0.4rem; margin-bottom: 1.2rem;">
  <div style="font-size: 2.0rem; font-weight: 800; line-height: 1.15;">{APP_TITLE}</div>
  <div style="margin-top: 0.35rem; color: #ffffff; opacity: 0.92;">
    Upload a PDF/DOCX/TXT and ask questions in a simple Q&A chat.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<div class='card'>", unsafe_allow_html=True)
u_left, u_right = st.columns([2, 1], gap="large")
with u_left:
    st.markdown("<div class='section-title'>Add a document</div>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>PDF, DOCX, TXT</div>", unsafe_allow_html=True)
    f = st.file_uploader(" ", type=["pdf", "docx", "txt"], label_visibility="collapsed")
    upload_clicked = st.button("Upload document", type="primary", disabled=not bool(f))
    if f and upload_clicked:
        files = {"file": (f.name, f.getvalue(), f.type or "application/octet-stream")}
        with st.spinner("Uploading and indexing…"):
            r = _http("POST", "/upload-document", files=files, timeout=240)
        if r.status_code != 200:
            st.error(r.text)
        else:
            data = r.json()
            doc_id = data["document_id"]
            st.session_state["last_upload"] = data
            st.session_state["active_doc_ids"] = [doc_id]
            existing = list(st.session_state["uploaded_docs"])
            existing.insert(0, UploadedDoc(document_id=doc_id, name=f.name, uploaded_at=_iso_now()))
            st.session_state["uploaded_docs"] = existing
            # New document → fresh chat for this page session.
            st.session_state["session_id"] = ""
            st.session_state["chat_messages"] = []
            st.success("Uploaded successfully.")
            st.markdown(
                f"<span class='small-muted'>Document ID</span><br/><span class='pill mono'>{doc_id}</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"Chunks created: {data.get('chunks_created')}")

with u_right:
    active_doc_ids: list[str] = st.session_state.get("active_doc_ids") or []
    st.markdown("<div class='section-title'>Active document</div>", unsafe_allow_html=True)
    if active_doc_ids:
        st.markdown(f"<span class='pill mono'>{active_doc_ids[0]}</span>", unsafe_allow_html=True)
        docs: list[UploadedDoc] = st.session_state.get("uploaded_docs") or []
        if docs:
            st.markdown(f"<div class='small-muted'>File</div><div><b>{docs[0].name}</b></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='small-muted'>No document uploaded yet.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)


st.markdown("### Chat")
active_doc_ids = st.session_state.get("active_doc_ids") or []
if not active_doc_ids:
    st.info("Upload a document to start asking questions.")

# Render chat transcript
for m in st.session_state.get("chat_messages") or []:
    with st.chat_message(m["role"]):
        st.write(m["content"])

q = st.chat_input("Message your document…", disabled=not bool(active_doc_ids))
if q:
    st.session_state["chat_messages"].append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)

    payload = {
        "user_id": _user_id(),
        "session_id": (st.session_state.get("session_id") or "").strip() or None,
        "question": q,
        "doc_ids": active_doc_ids or None,
    }
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            r = _http("POST", "/ask-question", json=payload, timeout=300)
        if r.status_code != 200:
            st.error(r.text)
            st.session_state["chat_messages"].append({"role": "assistant", "content": f"Error: {r.text}"})
        else:
            data = r.json()
            st.session_state["session_id"] = data["session_id"]
            answer = data["answer"]
            st.write(answer)
            st.session_state["chat_messages"].append({"role": "assistant", "content": answer})

