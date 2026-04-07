from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from backend.common.config import settings


def _is_transient_genai_exception(e: Exception) -> bool:
    """
    Retry only on likely-transient failures (timeouts, 429s, 5xx).

    Do NOT retry "model not found / not supported" type errors; those are configuration problems.
    """
    msg = str(e).lower()
    if "404" in msg and ("not found" in msg or "is not found" in msg):
        return False
    if "not supported for embedcontent" in msg:
        return False

    status = getattr(e, "status_code", None)
    if isinstance(status, int):
        if status in (408, 409, 425, 429):
            return True
        if 500 <= status <= 599:
            return True

    if any(s in msg for s in ("timeout", "timed out", "temporar", "rate limit", "429", "503", "server error")):
        return True
    return False


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: list[list[float]]
    model: str


class GeminiClient:
    """
    Thin wrapper around the Google GenAI SDK, with retries and a stable interface
    for chat + embeddings.
    """

    def __init__(self) -> None:
        # Use the stable `v1` API for better model availability/consistency.
        self._client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(api_version="v1"),
        )
        # Resolved at runtime after the first successful call.
        self._resolved_embed_model: str | None = None

    def _is_model_not_found(self, e: Exception) -> bool:
        """
        The google-genai SDK error shape can vary by version.
        Detect "model not found / not supported" in a robust way.
        """
        msg = str(e).lower()
        if "404" in msg and ("not found" in msg or "is not found" in msg):
            return True
        if "not supported for embedcontent" in msg:
            return True
        # Best-effort access to structured fields when present.
        status = getattr(e, "status_code", None)
        if status == 404:
            return True
        return False

    def _is_transient_error(self, e: Exception) -> bool:
        """
        Decide whether an error is worth retrying.

        Important: "model not found / not supported" errors are NOT transient and should not be retried,
        otherwise we hammer the API with the same bad model repeatedly.
        """
        if self._is_model_not_found(e):
            return False

        # Best-effort checks across SDK versions.
        status = getattr(e, "status_code", None)
        if isinstance(status, int):
            if status in (408, 409, 425, 429):
                return True
            if 500 <= status <= 599:
                return True

        msg = str(e).lower()
        if any(s in msg for s in ("timeout", "timed out", "temporar", "rate limit", "429", "503", "server error")):
            return True
        return False

    def _candidate_embed_models(self, preferred: str | None) -> list[str]:
        """
        Return candidate models in the order we want to try.

        Model availability varies by API version / key.
        These IDs are commonly supported on the Gemini Developer API:
        - `gemini-embedding-001` (text)
        - `text-embedding-004` / `text-embedding-005` (text)
        - `gemini-embedding-2-preview` (multimodal; also works for text)
        """
        candidates: list[str] = []
        if preferred:
            candidates.append(preferred)
        for m in ("gemini-embedding-001", "text-embedding-004", "text-embedding-005", "gemini-embedding-2-preview"):
            if m not in candidates:
                candidates.append(m)
        return candidates

    @retry(
        wait=wait_exponential_jitter(initial=0.5, max=8),
        stop=stop_after_attempt(5),
        retry=retry_if_exception(_is_transient_genai_exception),
    )
    def embed_texts(self, texts: list[str], *, model: str | None = None) -> EmbeddingResult:
        """
        Embeddings can differ by endpoint/account availability.
        We try the configured model first, then fall back to known options.
        """
        if self._resolved_embed_model and not model:
            preferred = self._resolved_embed_model
        else:
            preferred = model or settings.gemini_embed_model

        candidates = self._candidate_embed_models(preferred)

        last_err: Exception | None = None
        for m in candidates:
            try:
                resp = self._client.models.embed_content(model=m, contents=texts)
                vectors = [e.values for e in resp.embeddings]
                if not model:
                    self._resolved_embed_model = m
                return EmbeddingResult(vectors=vectors, model=m)
            except genai_errors.ClientError as e:
                # If the model/method isn't supported, try the next candidate.
                if self._is_model_not_found(e):
                    last_err = e
                    continue
                raise
            except Exception as e:
                # Some SDK versions may raise a different exception type.
                if self._is_model_not_found(e):
                    last_err = e
                    continue
                raise

        assert last_err is not None
        raise last_err

    @retry(
        wait=wait_exponential_jitter(initial=0.5, max=8),
        stop=stop_after_attempt(5),
        retry=retry_if_exception(_is_transient_genai_exception),
        reraise=True,
    )
    def chat(self, prompt: str, *, model: str | None = None, temperature: float = 0.2) -> str:
        use_model = model or settings.gemini_chat_model
        resp = self._client.models.generate_content(
            model=use_model,
            contents=prompt,
            config={
                "temperature": temperature,
            },
        )
        return (resp.text or "").strip()

    @retry(
        wait=wait_exponential_jitter(initial=0.5, max=8),
        stop=stop_after_attempt(5),
        retry=retry_if_exception(_is_transient_genai_exception),
        reraise=True,
    )
    def chat_json(self, prompt: str, *, model: str | None = None, temperature: float = 0.0) -> Any:
        """
        Best-effort JSON response: asks Gemini for JSON and parses it.
        """
        import json

        text = self.chat(prompt, model=model, temperature=temperature)
        # Minimal robustness: strip code fences if present.
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[1] if "```" in cleaned else cleaned
        cleaned = cleaned.strip()
        return json.loads(cleaned)

