from __future__ import annotations

import json
from typing import Any

import httpx

from backend.common.config import settings


class OllamaClient:
    """
    Minimal Ollama client for text generation.

    Requires an Ollama server running locally (default: http://127.0.0.1:11434).
    """

    def __init__(self) -> None:
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._timeout_s = float(settings.ollama_timeout_seconds)

    def chat(self, prompt: str, *, model: str | None = None, temperature: float = 0.2) -> str:
        use_model = model or settings.ollama_model
        if not use_model:
            raise RuntimeError("OLLAMA_MODEL is not set")

        url = f"{self._base_url}/api/generate"
        payload = {
            "model": use_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        try:
            with httpx.Client(timeout=self._timeout_s) as client:
                resp = client.post(url, json=payload)
        except Exception as e:
            raise RuntimeError(
                f"Failed to reach Ollama at {self._base_url}. Is Ollama running? "
                f"Try: `ollama serve` and confirm `ollama list` shows the model."
            ) from e

        if resp.status_code >= 400:
            raise RuntimeError(f"Ollama error {resp.status_code}: {resp.text}")

        data = resp.json()
        text = (data.get("response") or "").strip()
        return text

    def chat_json(self, prompt: str, *, model: str | None = None, temperature: float = 0.0) -> Any:
        text = self.chat(prompt, model=model, temperature=temperature)
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[1] if "```" in cleaned else cleaned
        cleaned = cleaned.strip()
        return json.loads(cleaned)

