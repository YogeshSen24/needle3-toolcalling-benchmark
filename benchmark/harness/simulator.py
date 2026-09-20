"""Deterministic world simulator.

No network, no randomness, no clock reads. End-to-end task success is the final
world state matching the expected state, which is the only metric invariant to
tool granularity and to harmless differences in call order.
"""
from __future__ import annotations

from typing import Any

from ..adapters.base import ToolCall


def flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(node, list):
        out[prefix] = repr(node)
    else:
        out[prefix] = node
    return out


class Simulator:
    def __init__(self, env, profile, overrides: dict | None = None):
        self.env = env
        self.profile = profile
        self._exec = env.executors(profile)
        self.state = env.initial_state(overrides)
        self.trace: list[dict] = []

    def execute(self, call: ToolCall) -> dict:
        fn = self._exec.get(call.name)
        if fn is None:
            result = {"error": f"unknown tool: {call.name}"}
        else:
            try:
                result = fn(self.state, **(call.arguments or {}))
            except TypeError as exc:
                # Wrong or missing arguments are a measurable failure mode,
                # not a harness crash.
                result = {"error": f"bad arguments: {exc}"}
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
        self.trace.append({"call": call.to_dict(), "result": result})
        return result

    def flat_state(self) -> dict[str, Any]:
        return flatten(self.state)

    def diff_against(self, expected: dict[str, Any]) -> dict[str, dict]:
        flat = self.flat_state()
        return {
            path: {"expected": want, "actual": flat.get(path, "<missing>")}
            for path, want in (expected or {}).items()
            if flat.get(path, "<missing>") != want
        }
