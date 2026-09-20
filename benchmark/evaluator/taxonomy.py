"""Failure taxonomy (spec §17).

Every label here is derived mechanically from the comparison between expected
and actual calls. None of it is inferred by a model. Where a cause genuinely
cannot be established programmatically, no label is emitted rather than a
guessed one — the report calls anything model-proposed a "hypothesized failure
reason" and keeps it out of the counts.
"""
from __future__ import annotations

from collections import Counter

from ..adapters.base import ToolCall


def _key(call) -> tuple:
    from .scoring import _call_key
    return _call_key(call)


def classify(case, expected: list[dict], actual: list[ToolCall], score, read_only: set) -> list[str]:
    if score.full_call_exact and score.e2e_success:
        return []

    labels: list[str] = []
    exp_names = Counter(c["name"] for c in expected)
    act_names = Counter(c.name for c in actual)

    if score.error:
        labels.append("runtime_error")

    # Abstain-vs-act errors first: they dominate the interpretation.
    if not score.expected_action and score.took_action:
        labels.append("unsupported_but_called" if case.expected.behavior in ("no_call", "clarify")
                      else "failed_condition")
    if score.expected_action and not score.took_action:
        labels.append("missing_tool" if not actual else "failed_condition")

    missing = exp_names - act_names
    extra = act_names - exp_names
    if missing and actual:
        labels.append("missing_tool")
    if extra:
        # A wrong name where one was expected is a substitution, not an addition.
        labels.append("wrong_tool" if missing else "extra_tool")

    # Argument-level errors, only for tools that were selected correctly.
    shared = set(exp_names) & set(act_names)
    for name in shared:
        exp_args = [c["arguments"] for c in expected if c["name"] == name]
        act_args = [c.arguments for c in actual if c.name == name]
        for want, got in zip(exp_args, act_args):
            from .scoring import _norm_args
            want_n, got_n = _norm_args(want), _norm_args(got)
            if want_n == got_n:
                continue
            for key, value in want_n.items():
                if key not in got_n:
                    labels.append("missing_argument")
                elif got_n[key] != value:
                    labels.append("wrong_argument")
            for key in got_n:
                if key not in want_n:
                    labels.append("invented_argument")

    if case.expected.order_matters and not score.sequence_success and score.full_call_exact:
        labels.append("wrong_order")

    if case.properties.negation and score.took_action and not score.full_call_exact:
        labels.append("failed_negation")
    if case.properties.conditional and not score.full_call_exact:
        labels.append("failed_condition")
    if case.context.get("location") and any(
        "room" in (c.arguments or {}) and c.arguments.get("room") != case.context["location"]
        for c in actual
    ) and case.complexity_level == 3:
        labels.append("failed_context")
    if case.properties.ambiguous and score.took_action:
        labels.append("ambiguity_failure")
    if "below_confidence_threshold" in score.gated_out or "negation" in score.gated_out \
            or "ungrounded" in score.gated_out:
        if score.expected_action:
            labels.append("gated_by_contract")

    seen: set[str] = set()
    return [x for x in labels if not (x in seen or seen.add(x))]
