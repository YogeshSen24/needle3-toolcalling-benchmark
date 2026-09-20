"""Harness profiles: the optimization axes, as a factorial design.

The report's third chapter asks "how much can a good harness buy you?". That
question is only answerable if each lever is isolated. A profile is one point in
the configuration space; the experiment runs the baseline, then each lever alone,
then everything together, so every claim has a single-factor effect size behind
it rather than a bundle.

Every lever here is something an application developer can actually do without
retraining the model. Fine-tuning is explicitly out of scope: it would answer a
different question (see DESIGN.md §3).
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Profile:
    name: str

    # --- tool surface -----------------------------------------------------
    descriptions: str = "naive"
    """naive  = what a competent developer writes without reading vendor docs.
       tuned  = vendor house style: literal formats, spans named in prose,
                enum values chosen to match the words users actually say."""

    constraints: str = "loose"
    """loose = free strings and unbounded numbers.
       tight = closed enums, min/max bounds, patterns. These compile into the
               decode grammar, so an invalid value becomes unrepresentable
               rather than merely discouraged."""

    granularity: str = "verb_split"
    """verb_split  = one tool per action (turn_off_light).
       action_enum = one tool per device class with an action enum. The vendor's
                     shipped environments use the latter."""

    # --- request surface --------------------------------------------------
    context_mode: str = "system"
    """system = ambient facts go in the system turn under Needle's recognized
                keys.
       inline = the harness appends the same facts to the user turn. Phase 1
                measured 0/5 vs 4/5 for exactly this change, because arguments
                must be grounded in the request."""

    normalize_prompt: bool = False
    """Deterministic, model-free rewriting of the user turn (spelled-out numbers
       to digits, and similar). Never an LLM: that would smuggle a second model
       into a single-model benchmark."""

    # --- routing ----------------------------------------------------------
    triggers: bool = False
    """Attach regex triggers to tools. A matching trigger restricts decoding to
       the matched tools and forces a call, bypassing the confidence threshold."""

    loop_breaker: bool = False
    """Stop the agentic loop when the model re-emits a call it already made.
       The baseline run showed 42 of 110 episodes continuing past completion,
       usually repeating one call until the step limit, which executes the
       same side effect several times."""

    prefilter: int | None = None
    """Pre-select the top-k tools by embedding similarity (agent.embed) and
       expose only those. Needle already retrieves internally above five tools;
       this tests whether doing it explicitly, with our own query, beats the
       built-in path."""

    def label(self) -> str:
        return self.name

    def with_(self, **kwargs) -> "Profile":
        return replace(self, **kwargs)


# The baseline: a developer wires up tools the obvious way and passes ambient
# state in the system prompt, as almost every agent framework encourages.
BASELINE = Profile(name="baseline")

# Single-factor arms. Each differs from BASELINE in exactly one field.
SINGLE_FACTOR = [
    BASELINE.with_(name="opt_descriptions", descriptions="tuned"),
    BASELINE.with_(name="opt_constraints", constraints="tight"),
    BASELINE.with_(name="opt_context_inline", context_mode="inline"),
    BASELINE.with_(name="opt_normalize", normalize_prompt=True),
    BASELINE.with_(name="opt_triggers", triggers=True),
    BASELINE.with_(name="opt_loop_breaker", loop_breaker=True),
    BASELINE.with_(name="opt_prefilter5", prefilter=5),
    BASELINE.with_(name="opt_action_enum", granularity="action_enum"),
]

# Everything except granularity, which is reported separately because it is a
# redesign of the tool surface rather than a tuning knob.
OPTIMIZED = Profile(
    name="optimized_all",
    descriptions="tuned",
    constraints="tight",
    context_mode="inline",
    normalize_prompt=True,
    triggers=True,
    loop_breaker=True,
    prefilter=5,
)

# Only the levers that measured positive or free in the single-factor run.
# opt_prefilter5 is excluded because it measured significantly *negative*, and
# stacking a harmful lever just to say "we applied everything" would hide that.
OPTIMIZED_SELECTED = Profile(
    name="optimized_selected",
    descriptions="tuned",
    constraints="tight",
    context_mode="inline",
    loop_breaker=True,
)

ALL_PROFILES = [BASELINE, *SINGLE_FACTOR, OPTIMIZED, OPTIMIZED_SELECTED]
BY_NAME = {p.name: p for p in ALL_PROFILES}


def get(name: str) -> Profile:
    if name not in BY_NAME:
        raise KeyError(f"unknown profile {name!r}; have {sorted(BY_NAME)}")
    return BY_NAME[name]
