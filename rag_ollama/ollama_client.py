from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


class OllamaError(RuntimeError):
    pass


@dataclass(frozen=True)
class OllamaClient:
    base_url: str = "http://localhost:11434"
    timeout_s: float = 120.0

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    def tags(self) -> Dict[str, Any]:
        try:
            resp = requests.get(self._url("/api/tags"), timeout=self.timeout_s)
        except requests.RequestException as e:
            raise OllamaError(
                f"Failed to reach Ollama at {self.base_url}. Is the Docker container running and port 11434 exposed?"
            ) from e
        if resp.status_code != 200:
            raise OllamaError(f"Ollama /api/tags failed: {resp.status_code} {resp.text}")
        return resp.json()

    def embeddings(self, *, model: str, prompt: str) -> list[float]:
        payload = {"model": model, "prompt": prompt}
        try:
            resp = requests.post(self._url("/api/embeddings"), json=payload, timeout=self.timeout_s)
        except requests.RequestException as e:
            raise OllamaError(f"Ollama /api/embeddings request failed: {e}") from e
        if resp.status_code != 200:
            raise OllamaError(f"Ollama /api/embeddings failed: {resp.status_code} {resp.text}")
        data = resp.json()
        emb = data.get("embedding")
        if not isinstance(emb, list) or not emb:
            raise OllamaError(f"Unexpected embeddings response: {data}")
        return emb

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options

        try:
            resp = requests.post(self._url("/api/generate"), json=payload, timeout=self.timeout_s)
        except requests.RequestException as e:
            raise OllamaError(f"Ollama /api/generate request failed: {e}") from e
        if resp.status_code != 200:
            raise OllamaError(f"Ollama /api/generate failed: {resp.status_code} {resp.text}")
        data = resp.json()
        txt = data.get("response")
        if not isinstance(txt, str):
            raise OllamaError(f"Unexpected generate response: {data}")
        return txt
