"""Scoring contracts (DESIGN.md F6, §C.1).

Needle's raw output and the contract its own shipped harness applies disagree
on real cases. Observed during design: a correct `set_light_brightness` call
carrying `validation.negation: true`, which the vendor harness converts into a
refusal. Reporting either number alone would misrepresent the model, in
opposite directions. So every metric is computed twice.

For models that expose neither `validation` nor a calibrated confidence, the
PRODUCTION contract reduces to RAW. That is stated in the report rather than
hidden, because it means the contract comparison is informative for Needle only.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..adapters.base import ModelResponse, ToolCall


@dataclass(frozen=True)
class Contract:
    name: str
    confidence_threshold: float | None = None
    apply_validation_gates: bool = False

    def gate(self, response: ModelResponse) -> tuple[list[ToolCall], str | None]:
        """Return the calls that would actually be executed, and why any were dropped."""
        calls = list(response.calls)
        if not calls:
            return calls, None

        if self.apply_validation_gates:
            validation = response.validation or {}
            if validation.get("ungrounded"):
                return [], "ungrounded"
            if validation.get("negation"):
                return [], "negation"

        if self.confidence_threshold is not None and response.confidence is not None:
            if response.confidence < self.confidence_threshold:
                return [], "below_confidence_threshold"

        return calls, None


RAW = Contract(name="raw")

# Mirrors needle/environments/_harness.py: drop on ungrounded or negation, then
# apply the confidence gate the vendor calls "the production contract".
PRODUCTION = Contract(name="production", confidence_threshold=0.4,
                      apply_validation_gates=True)


def production_at(threshold: float) -> Contract:
    """Used by the hybrid-routing sweep."""
    return Contract(name=f"production@{threshold:g}", confidence_threshold=threshold,
                    apply_validation_gates=True)


DEFAULT_CONTRACTS = [RAW, PRODUCTION]
