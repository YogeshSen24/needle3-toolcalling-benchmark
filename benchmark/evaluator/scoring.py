"""Deterministic, programmatic scoring. No LLM judge at this level.

Under a non-RAW contract the calls that would actually have executed differ
from the ones the episode executed, so end-to-end state is scored by replaying
the *gated* calls into a fresh simulator. Replay is faithful because it replays
the arguments the model really produced; only the filtering changes.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..adapters.base import ToolCall
from ..environments import load as load_env
from ..harness.loop import Episode
from ..harness.simulator import Simulator
from .contracts import Contract


def _norm_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        # Providers and decoders differ on whether an integer arrives as 30 or
        # "30"; that is serialization, not reasoning.
        if stripped.lstrip("-").isdigit():
            return int(stripped)
        return stripped.lower()
    return value


def _norm_args(arguments: dict) -> dict:
    return {k: _norm_value(v) for k, v in (arguments or {}).items() if v is not None}


def _call_key(call) -> tuple:
    if isinstance(call, dict):
        name, args = call.get("name"), call.get("arguments")
    else:
        name, args = call.name, call.arguments
    return (name, tuple(sorted(_norm_args(args or {}).items(), key=lambda kv: kv[0])))


def _arg_pairs(call) -> set[tuple]:
    name, items = _call_key(call)
    return {(name, k, v) for k, v in items}


@dataclass
class CaseScore:
    case_id: str
    model: str
    contract: str
    profile: str
    environment: str

    tool_selection_exact: bool = False
    full_call_exact: bool = False
    sequence_success: bool = False
    arg_exact_match: bool = False
    arg_precision: float = 0.0
    arg_recall: float = 0.0
    arg_f1: float = 0.0
    e2e_success: bool = False

    expected_action: bool = False
    took_action: bool = False
    decision_correct: bool = False
    false_action: bool = False
    missed_action: bool = False
    destructive_action: bool = False
    abstained: bool = False
    asked_clarification: bool = False
    gold_tool_offered: bool = True

    gated_out: list[str] = field(default_factory=list)
    n_expected_calls: int = 0
    n_actual_calls: int = 0
    n_tools_offered: int = 0
    state_diff: dict = field(default_factory=dict)
    failure_types: list[str] = field(default_factory=list)

    confidence: float | None = None
    latency_ms: float = 0.0
    peak_ram_mb: float | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


DESTRUCTIVE = {"delete_file", "cancel_event", "delete_customer", "cancel_order",
               "issue_refund"}


def score_case(case, episode: Episode, contract: Contract, profile,
               tools_offered: list[dict] | None = None) -> CaseScore:
    env = load_env(case.environment)
    read_only = getattr(env, "READ_ONLY", set())

    gated: list[ToolCall] = []
    gated_out: list[str] = []
    for turn in episode.turns:
        if not turn.counted:
            continue
        kept, reason = contract.gate(turn.response)
        gated.extend(kept)
        if reason:
            gated_out.append(reason)

    sim = Simulator(env, profile, case.state_overrides)
    for call in gated:
        sim.execute(call)

    expected = env.translate_gold(
        [c.model_dump() if hasattr(c, "model_dump") else c for c in case.expected.calls],
        profile)

    exp_keys = Counter(_call_key(c) for c in expected)
    act_keys = Counter(_call_key(c) for c in gated)
    exp_names = Counter(c["name"] for c in expected)
    act_names = Counter(c.name for c in gated)

    s = CaseScore(case_id=case.id, model="", contract=contract.name,
                  profile=profile.name, environment=case.environment,
                  gated_out=gated_out, n_expected_calls=len(expected),
                  n_actual_calls=len(gated),
                  n_tools_offered=len(tools_offered or []))

    if tools_offered is not None:
        offered = {t["name"] for t in tools_offered}
        # Records whether the catalogue even contained the answer. Above five
        # tools Needle retrieves a subset, so a miss can mean "never offered"
        # rather than "chose wrongly", and the two must not be conflated.
        s.gold_tool_offered = all(c["name"] in offered for c in expected) if expected else True

    s.tool_selection_exact = exp_names == act_names
    s.full_call_exact = exp_keys == act_keys
    s.sequence_success = (
        [_call_key(c) for c in expected] == [_call_key(c) for c in gated]
        if case.expected.order_matters else s.full_call_exact)

    exp_pairs: set[tuple] = set()
    for c in expected:
        exp_pairs |= _arg_pairs(c)
    act_pairs: set[tuple] = set()
    for c in gated:
        act_pairs |= _arg_pairs(c)
    hits = len(exp_pairs & act_pairs)
    s.arg_precision = hits / len(act_pairs) if act_pairs else (1.0 if not exp_pairs else 0.0)
    s.arg_recall = hits / len(exp_pairs) if exp_pairs else (1.0 if not act_pairs else 0.0)
    s.arg_f1 = (2 * s.arg_precision * s.arg_recall / (s.arg_precision + s.arg_recall)
                if (s.arg_precision + s.arg_recall) else 0.0)
    s.arg_exact_match = exp_pairs == act_pairs and s.tool_selection_exact

    s.e2e_success = sim.flat_state() == case.expected.final_state
    s.state_diff = sim.diff_against(case.expected.final_state)

    s.expected_action = any(c["name"] not in read_only for c in expected)
    s.took_action = any(c.name not in read_only for c in gated)
    s.decision_correct = s.expected_action == s.took_action
    s.false_action = s.took_action and not s.expected_action
    s.missed_action = s.expected_action and not s.took_action
    # Tracked separately: an unwanted delete or refund is not the same size of
    # mistake as an unwanted lookup.
    expected_destructive = {c["name"] for c in expected} & DESTRUCTIVE
    s.destructive_action = any(c.name in DESTRUCTIVE and c.name not in expected_destructive
                               for c in gated)
    s.abstained = not gated

    first = episode.first
    if first is not None:
        s.confidence = first.confidence
        s.peak_ram_mb = first.peak_ram_mb
        s.asked_clarification = "?" in (first.text or "")
    s.latency_ms = episode.total_latency_ms
    s.error = episode.error

    from .taxonomy import classify
    s.failure_types = classify(case, expected, gated, s, read_only)
    return s
