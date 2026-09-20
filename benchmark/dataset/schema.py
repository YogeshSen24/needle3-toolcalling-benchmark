"""Case schema.

Pydantic is a gate, not decoration: a malformed case must fail at build time,
not become a mysterious zero in a results table.

Difficulty ladder, within every environment:

    D0  direct          the request names the action and all its arguments
    D1  paraphrase      same, worded nothing like the tool
    D2  distraction     negation, irrelevant detail, or nothing applicable
    D3  context         an argument is ambient, not stated
    D4  parallel        two independent actions in one request
    D5  dependent       an argument, or whether to act at all, comes from a
                        previous call's result
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Behavior = Literal["tool_call", "multiple_tool_calls", "no_call", "clarify"]

LEVEL_NAMES = {0: "direct", 1: "paraphrase", 2: "distraction", 3: "context",
               4: "parallel", 5: "dependent"}


class ExpectedCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Expected(BaseModel):
    behavior: Behavior
    calls: list[ExpectedCall] = Field(default_factory=list)
    order_matters: bool = False
    final_state: dict[str, Any] = Field(default_factory=dict)

    @field_validator("calls")
    @classmethod
    def _no_calls_when_abstaining(cls, v, info):
        if info.data.get("behavior") in ("no_call", "clarify") and v:
            raise ValueError("abstain behaviours must not declare expected calls")
        return v


class Properties(BaseModel):
    negation: bool = False
    multi_tool: bool = False
    ambiguous: bool = False
    conditional: bool = False
    needs_ambient_context: bool = False
    adversarial_type: str | None = None
    pair_id: str | None = None
    conditional_branch: str | None = None
    language_variant: str = "canonical"
    source_case: str | None = None


class Provenance(BaseModel):
    author: str = "hand"
    contamination_checked: bool = False
    max_shipped_similarity: float | None = None


class Case(BaseModel):
    id: str
    split: Literal["dev", "holdout"] = "dev"
    environment: str
    tier: int
    complexity_level: int
    prompt: str
    context: dict[str, Any] = Field(default_factory=dict)
    state_overrides: dict[str, Any] = Field(default_factory=dict)
    expected: Expected
    properties: Properties = Field(default_factory=Properties)
    provenance: Provenance = Field(default_factory=Provenance)

    @property
    def level_name(self) -> str:
        return LEVEL_NAMES.get(self.complexity_level, str(self.complexity_level))
