"""The shared agentic loop.

Every model is driven through exactly this loop. Using each vendor's own agent
runner (needle's `run()`, an SDK's tool-runner) would confound loop policy with
model capability: different step limits, different grounding gates, different
result formats. The only thing allowed to differ per vendor is how tool results
are handed back, which each adapter does natively.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..adapters.base import ModelAdapter, ModelResponse, ToolCall
from .simulator import Simulator


@dataclass
class Turn:
    index: int
    response: ModelResponse
    executed: list[dict] = field(default_factory=list)
    counted: bool = True
    """False when the loop breaker rejected this turn. Scoring skips it: the
    calls were never executed, so counting them would score a side effect that
    never happened."""


@dataclass
class Episode:
    turns: list[Turn] = field(default_factory=list)
    calls: list[ToolCall] = field(default_factory=list)   # flattened, in order
    final_state: dict = field(default_factory=dict)
    total_latency_ms: float = 0.0
    steps: int = 0
    hit_step_limit: bool = False
    stopped_on_repeat: bool = False
    error: str | None = None

    @property
    def first(self) -> ModelResponse | None:
        return self.turns[0].response if self.turns else None

    def to_dict(self) -> dict:
        return {
            "turns": [{"index": t.index, "response": t.response.to_dict(),
                       "executed": t.executed, "counted": t.counted}
                      for t in self.turns],
            "calls": [c.to_dict() for c in self.calls],
            "final_state": self.final_state,
            "total_latency_ms": self.total_latency_ms,
            "steps": self.steps,
            "hit_step_limit": self.hit_step_limit,
            "stopped_on_repeat": self.stopped_on_repeat,
            "error": self.error,
        }


def run_episode(adapter: ModelAdapter, prompt: str, tools: list[dict], system: str,
                simulator: Simulator, max_steps: int = 4,
                loop_breaker: bool = False) -> Episode:
    episode = Episode()
    seen: set[str] = set()
    adapter.begin_episode(tools, system)
    try:
        response = adapter.send(prompt)
        for step in range(max_steps):
            episode.steps = step + 1
            episode.total_latency_ms += response.latency_ms
            turn = Turn(index=step, response=response)
            episode.turns.append(turn)

            if response.error:
                episode.error = response.error
                break
            if not response.calls:
                break

            # Terminate on a repeat rather than executing the same side effect
            # again. Checked before execution so nothing is applied twice.
            if loop_breaker and all(call.key() in seen for call in response.calls):
                episode.stopped_on_repeat = True
                turn.counted = False
                break

            results: list[tuple[str, Any]] = []
            for call in response.calls:
                seen.add(call.key())
                episode.calls.append(call)
                result = simulator.execute(call)
                turn.executed.append({"call": call.to_dict(), "result": result})
                results.append((call.name, result))

            if step == max_steps - 1:
                episode.hit_step_limit = True
                break
            response = adapter.send_results(results)
    finally:
        adapter.end_episode()
    episode.final_state = simulator.flat_state()
    return episode
