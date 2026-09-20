"""Shared case construction.

Expected final state is never typed by hand. It is produced by executing the
gold calls against the simulator, so ground truth cannot drift away from the
world model, and a gold call that the simulator rejects fails the build loudly
instead of silently becoming an unreachable target.
"""
from __future__ import annotations

from ..adapters.base import ToolCall
from ..environments import load as load_env
from ..harness.simulator import Simulator
from ..profiles import BASELINE
from .schema import Case


def call(name: str, **arguments) -> dict:
    return {"name": name, "arguments": arguments}


def derive_final_state(env_id: str, overrides: dict, gold: list[dict]) -> dict:
    env = load_env(env_id)
    sim = Simulator(env, BASELINE, overrides)
    for c in gold:
        result = sim.execute(ToolCall.from_dict(c))
        if isinstance(result, dict) and result.get("error"):
            raise ValueError(f"gold call rejected by simulator: {c} -> {result}")
    return sim.flat_state()


def build(env_id: str, tier: int, rows: list[tuple]) -> list[Case]:
    """rows: (suffix, level, prompt, context, overrides, behavior, gold, props)"""
    cases: list[Case] = []
    for suffix, level, prompt, ctx, overrides, behavior, gold, props in rows:
        cases.append(Case(
            id=f"{env_id}.D{level}.{suffix}",
            environment=env_id,
            tier=tier,
            complexity_level=level,
            prompt=prompt,
            context=ctx,
            state_overrides=overrides,
            expected={
                "behavior": behavior,
                "calls": gold,
                # Order only matters when a later call consumes an earlier result.
                "order_matters": level >= 5,
                "final_state": derive_final_state(env_id, overrides, gold),
            },
            properties=props,
        ))
    return cases
