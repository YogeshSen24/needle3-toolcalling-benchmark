"""Model registry: every model ID comes from config or environment, never code."""
from __future__ import annotations

import os
from pathlib import Path

from .base import ModelAdapter


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader. Existing environment variables always win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def build_adapter(spec: dict) -> ModelAdapter:
    """Construct an adapter from a config block.

    spec: {"adapter": "needle3"|"gemini", "model": <id>, ...}
    """
    kind = spec.get("adapter")
    if kind == "needle3":
        from .needle3 import Needle3Adapter
        return Needle3Adapter(
            max_new_tokens=int(spec.get("max_new_tokens", 512)),
            weights=spec.get("weights"),
        )
    if kind == "gemini":
        from .gemini import GeminiAdapter
        model = spec.get("model") or os.environ.get(spec.get("model_env", ""), "")
        if not model:
            raise RuntimeError(
                f"No model id for '{spec.get('name')}'. Set it in config/models.yaml "
                f"or via {spec.get('model_env')} in .env"
            )
        return GeminiAdapter(
            model=model,
            temperature=float(spec.get("temperature", 0.0)),
            max_tokens=int(spec.get("max_tokens", 1024)),
            name=spec.get("name"),
        )
    raise ValueError(f"unknown adapter kind: {kind!r}")
