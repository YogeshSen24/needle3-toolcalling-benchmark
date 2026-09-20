"""Normalized adapter interface.

Nothing downstream of this module may touch a vendor-specific response shape.
Scoring, taxonomy and reporting all consume `ModelResponse` only, which is what
keeps the comparison honest: a field that one vendor happens to expose cannot
leak into another model's score.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]

    def key(self) -> str:
        return self.name + "|" + json.dumps(self.arguments, sort_keys=True, default=str)

    def to_dict(self) -> dict:
        return {"name": self.name, "arguments": self.arguments}

    @staticmethod
    def from_dict(d: dict) -> "ToolCall":
        return ToolCall(str(d.get("name")), dict(d.get("arguments") or {}))


@dataclass
class ModelResponse:
    """One model turn, normalized across vendors.

    Fields that only some vendors provide stay None rather than being imputed;
    an absent measurement must never look like a measured zero.
    """
    calls: list[ToolCall] = field(default_factory=list)
    text: str | None = None
    confidence: float | None = None
    latency_ms: float = 0.0

    # Needle-specific, recorded but never used as a score (spec §3).
    suppressed_calls: list[dict] = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    reasoning: str | None = None
    prefill_tps: float | None = None
    decode_tps: float | None = None
    peak_ram_mb: float | None = None

    # Remote-model accounting.
    input_tokens: int | None = None
    output_tokens: int | None = None

    error: str | None = None
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "calls": [c.to_dict() for c in self.calls],
            "text": self.text,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "suppressed_calls": self.suppressed_calls,
            "validation": self.validation,
            "reasoning": self.reasoning,
            "prefill_tps": self.prefill_tps,
            "decode_tps": self.decode_tps,
            "peak_ram_mb": self.peak_ram_mb,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error": self.error,
            "raw": self.raw,
        }


class ModelAdapter(ABC):
    """Stateful, single-episode-at-a-time adapter.

    The runner drives every model through the identical sequence:
        begin_episode(tools, system) -> send(prompt) -> [send_results(...)]* -> end_episode()
    """

    name: str = "unnamed"
    concurrency_safe: bool = True

    @abstractmethod
    def begin_episode(self, tools: list[dict], system: str) -> None:
        """Bind a toolset and system-fact string, clearing conversation state."""

    @abstractmethod
    def send(self, user_text: str) -> ModelResponse:
        """Send a user turn."""

    @abstractmethod
    def send_results(self, results: list[tuple[str, Any]]) -> ModelResponse:
        """Feed executed tool results back in this vendor's *native* form.

        Deliberately not unified: forcing one wire format on every vendor would
        measure format mismatch rather than capability.
        """

    def end_episode(self) -> None:
        return None

    def close(self) -> None:
        return None

    def describe(self) -> dict:
        """Provenance for results/environment.json."""
        return {"name": self.name}
