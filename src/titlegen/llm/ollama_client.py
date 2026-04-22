from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class OllamaClient:
    host: str
    model: str
    timeout_seconds: int = 180

    def generate(self, prompt: str, options: Dict[str, Any] | None = None) -> str:
        url = self.host.rstrip("/") + "/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            payload["options"] = options

        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot reach Ollama at {url}: {exc}") from exc

        parsed = json.loads(body)
        if "response" not in parsed:
            raise RuntimeError(f"Unexpected Ollama response: {parsed}")

        return str(parsed["response"]).strip()