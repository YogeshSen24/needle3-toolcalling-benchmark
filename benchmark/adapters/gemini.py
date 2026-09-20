"""Google Gemini adapter using native function calling over the REST API.

Deliberately dependency-free (urllib, not the SDK) so the wire payload is fully
visible and auditable: the parity rule in DESIGN.md §A.2 only means something if
we can see exactly what each model received.

Tool schemas arrive here in the same JSON-Schema form Needle receives and are
translated structurally, never rewritten or enriched.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .base import ModelAdapter, ModelResponse, ToolCall

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_LIST_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"

# Gemini's schema dialect rejects some JSON-Schema keywords outright.
_SCHEMA_ALLOWED = {
    "type", "format", "description", "nullable", "enum", "items",
    "properties", "required", "minimum", "maximum",
}


def _clean_schema(node: Any) -> Any:
    """Structurally convert JSON Schema to Gemini's subset.

    Dropped keywords are constraints Gemini cannot express. That is a real
    asymmetry with Needle's grammar-constrained decoding, so the runner records
    which keywords were dropped rather than hiding the difference.
    """
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key not in _SCHEMA_ALLOWED:
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {k: _clean_schema(v) for k, v in value.items()}
        elif key == "items":
            out[key] = _clean_schema(value)
        elif key == "type" and isinstance(value, str):
            out[key] = value.upper()
        else:
            out[key] = value
    return out


def dropped_keywords(tools: list[dict]) -> list[str]:
    """Schema keywords Gemini cannot represent, for the fairness appendix."""
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key not in _SCHEMA_ALLOWED and key not in ("name", "parameters", "description"):
                    found.add(key)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for tool in tools:
        walk(tool.get("parameters", {}))
    return sorted(found)


class GeminiAdapter(ModelAdapter):
    concurrency_safe = True

    def __init__(self, model: str, api_key: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 1024, timeout: float = 120.0, max_retries: int = 4,
                 name: str | None = None):
        self.model = model
        self.name = name or f"gemini:{model}"
        self._key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self._key:
            raise RuntimeError(
                "No Gemini API key. Set GEMINI_API_KEY in the environment or in .env"
            )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self._declarations: list[dict] = []
        self._system: str = ""
        self._contents: list[dict] = []
        self._dropped: list[str] = []

    # -- episode lifecycle ------------------------------------------------

    def begin_episode(self, tools: list[dict], system: str) -> None:
        self._declarations = [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": _clean_schema(t.get("parameters", {"type": "object", "properties": {}})),
            }
            for t in tools
        ]
        self._dropped = dropped_keywords(tools)
        self._system = system
        self._contents = []

    def send(self, user_text: str) -> ModelResponse:
        self._contents.append({"role": "user", "parts": [{"text": user_text}]})
        return self._generate()

    def send_results(self, results: list[tuple[str, Any]]) -> ModelResponse:
        # Gemini's native tool-result form: one functionResponse part per call.
        parts = [
            {"functionResponse": {"name": name, "response": _wrap(result)}}
            for name, result in results
        ]
        self._contents.append({"role": "user", "parts": parts})
        return self._generate()

    def end_episode(self) -> None:
        self._contents = []

    # -- core -------------------------------------------------------------

    def _generate(self) -> ModelResponse:
        body: dict[str, Any] = {
            "contents": self._contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }
        if self._system:
            body["systemInstruction"] = {"parts": [{"text": self._system}]}
        if self._declarations:
            body["tools"] = [{"functionDeclarations": self._declarations}]
            # AUTO, never ANY: forcing a call would destroy every no_call case.
            body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

        t0 = time.perf_counter()
        try:
            payload = self._post(body)
        except Exception as exc:
            return ModelResponse(
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        calls: list[ToolCall] = []
        texts: list[str] = []
        candidates = payload.get("candidates") or []
        model_parts: list[dict] = []
        if candidates:
            model_parts = (candidates[0].get("content") or {}).get("parts") or []
        for part in model_parts:
            if "functionCall" in part:
                fc = part["functionCall"]
                calls.append(ToolCall(str(fc.get("name")), dict(fc.get("args") or {})))
            elif "text" in part:
                texts.append(part["text"])

        # Preserve the assistant turn so multi-step episodes stay coherent.
        if model_parts:
            self._contents.append({"role": "model", "parts": model_parts})

        usage = payload.get("usageMetadata") or {}
        return ModelResponse(
            calls=calls,
            text="\n".join(texts).strip() or None,
            confidence=None,  # Gemini exposes no calibrated confidence
            latency_ms=latency_ms,
            input_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
            raw=payload,
        )

    def _post(self, body: dict) -> dict:
        url = _ENDPOINT.format(model=self.model)
        data = json.dumps(body).encode("utf-8")
        last: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json", "x-goog-api-key": self._key},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:500]
                last = RuntimeError(f"HTTP {exc.code}: {detail}")
                if exc.code in (408, 429, 500, 502, 503, 504):
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise last from exc
            except urllib.error.URLError as exc:
                last = RuntimeError(f"network: {exc}")
                time.sleep(min(2 ** attempt, 30))
        raise last or RuntimeError("request failed")

    def describe(self) -> dict:
        return {
            "name": self.name,
            "provider": "google-gemini",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "schema_keywords_dropped": self._dropped,
        }


def _wrap(result: Any) -> dict:
    return result if isinstance(result, dict) else {"result": result}


def list_models(api_key: str | None = None) -> list[dict]:
    """Live model list, so model IDs are verified rather than guessed."""
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("No Gemini API key in environment")
    req = urllib.request.Request(_LIST_ENDPOINT + "?pageSize=200",
                                 headers={"x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    out = []
    for m in payload.get("models", []):
        out.append({
            "id": m.get("name", "").replace("models/", ""),
            "display": m.get("displayName"),
            "input_limit": m.get("inputTokenLimit"),
            "methods": m.get("supportedGenerationMethods", []),
        })
    return out
