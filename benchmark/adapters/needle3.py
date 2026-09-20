"""Needle 3 adapter (cactus-needle).

Two runtime facts drive this file, both verified against 3.0.2 (DESIGN.md F2, F3):

* The C engine is a process-global singleton: `needle._active[generation] = self`.
  Constructing a second Needle steals the binding, so this adapter declares
  `concurrency_safe = False` and the runner must keep concurrency at 1.
* Output is deterministic, so a single run per case is sufficient. The adapter
  does not enforce that; the runner decides repeats from config.

The adapter captures the entire response envelope verbatim in `raw`, including
confidence, which is recorded but never used as a score.
"""
from __future__ import annotations

import json
import time
from collections import OrderedDict
from typing import Any

from .base import ModelAdapter, ModelResponse, ToolCall

_MAX_CACHED_AGENTS = 8


class Needle3Adapter(ModelAdapter):
    name = "needle3"
    concurrency_safe = False

    def __init__(self, max_new_tokens: int = 512, weights: str | None = None):
        import needle  # imported lazily so the package is optional for LLM-only runs

        self._needle = needle
        self._max_new_tokens = max_new_tokens
        self._weights = weights
        self._agents: OrderedDict[str, Any] = OrderedDict()
        self._agent = None
        self._init_seconds: list[float] = []
        self.rebinds = 0

    # -- episode lifecycle ------------------------------------------------

    def begin_episode(self, tools: list[dict], system: str) -> None:
        key = json.dumps([tools, system], sort_keys=True, default=str)
        agent = self._agents.get(key)
        if agent is None:
            t0 = time.perf_counter()
            agent = self._needle.Needle(
                tools=tools,
                system=system,
                weights=self._weights,
                # auto_date would inject *today's* date and make cases
                # irreproducible; the canonical context always supplies one.
                auto_date=False,
            )
            self._init_seconds.append(time.perf_counter() - t0)
            self._agents[key] = agent
            if len(self._agents) > _MAX_CACHED_AGENTS:
                _, evicted = self._agents.popitem(last=False)
                try:
                    evicted.close()
                except Exception:
                    pass
        else:
            self._agents.move_to_end(key)
        if agent is not self._agent:
            self.rebinds += 1
        self._agent = agent
        agent.reset()

    def send(self, user_text: str) -> ModelResponse:
        return self._complete(user_text)

    def send_results(self, results: list[tuple[str, Any]]) -> ModelResponse:
        # Mirrors needle.Needle.run(): results return as a JSON user turn.
        payload = json.dumps([r for _, r in results], default=str)
        return self._complete(payload)

    def end_episode(self) -> None:
        self._agent = None

    def close(self) -> None:
        for agent in self._agents.values():
            try:
                agent.close()
            except Exception:
                pass
        self._agents.clear()
        self._agent = None

    # -- core -------------------------------------------------------------

    def _complete(self, text: str) -> ModelResponse:
        if self._agent is None:
            raise RuntimeError("begin_episode() must be called before send()")
        t0 = time.perf_counter()
        try:
            raw = self._agent.complete(text, max_new_tokens=self._max_new_tokens)
        except Exception as exc:  # engine faults are data, not crashes
            return ModelResponse(
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        calls = [ToolCall.from_dict(c) for c in (raw.get("function_calls") or [])]
        return ModelResponse(
            calls=calls,
            # Needle emits no assistant prose; `reason` is an engine field, not a reply.
            text=None,
            confidence=raw.get("confidence"),
            latency_ms=latency_ms,
            suppressed_calls=list(raw.get("suppressed_calls") or []),
            validation=dict(raw.get("validation") or {}),
            reasoning=raw.get("reasoning"),
            prefill_tps=raw.get("prefill_tps"),
            decode_tps=raw.get("decode_tps"),
            peak_ram_mb=raw.get("peak_ram_mb"),
            error=raw.get("error"),
            raw=raw,
        )

    # -- diagnostics -------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Used for retrieval recall@5 diagnostics and the contamination gate."""
        if self._agent is None:
            raise RuntimeError("begin_episode() must be called before embed()")
        return self._agent.embed(text)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "package": "cactus-needle",
            "version": getattr(self._needle, "__version__", None),
            "generation": 3,
            "weights": self._weights or "base",
            "max_new_tokens": self._max_new_tokens,
            "mean_init_seconds": (
                sum(self._init_seconds) / len(self._init_seconds)
                if self._init_seconds else None
            ),
            "concurrency_safe": self.concurrency_safe,
        }
