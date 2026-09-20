"""Three environments, ordered by how hard they are to route in.

    T1 smart_home    5-10 tools, disjoint verbs, closed enums, one entity type
    T2 workspace     ~12 tools, shared verbs across entity types, ids and dates
    T3 business_ops  ~20 tools, deliberately confusable names, several entity
                     types that appear in the same sentence

The tiers are not "easy / medium / hard" as a label. They differ on properties
that can be counted: tool count, how many tools share a leading verb, mean
pairwise description similarity, arguments per call, and whether an argument
must be carried from one call to the next. `describe_tier()` reports those
numbers so the tiering is auditable rather than asserted.

Every environment exposes the same interface:

    ENV_ID, TIER, initial_state(overrides) -> dict
    tools(profile) -> list[schema]
    executors(profile) -> {name: callable(state, **args)}
    READ_ONLY: set[str]
    translate_gold(calls, profile) -> list[call]   (identity unless granularity moves)
"""
from __future__ import annotations

import importlib

_NAMES = ("smart_home", "workspace", "business_ops")


def load(name: str):
    if name not in _NAMES:
        raise KeyError(f"unknown environment {name!r}; have {list(_NAMES)}")
    return importlib.import_module(f"{__name__}.{name}")


def all_environments() -> dict:
    return {name: load(name) for name in _NAMES}


def _token_set(text: str) -> set[str]:
    import re
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def describe_tier(module, profile) -> dict:
    """Countable properties behind the tier ordering."""
    schemas = module.tools(profile)
    names = [s["name"] for s in schemas]
    verbs = [n.split("_")[0] for n in names]
    shared_verb = len(names) - len(set(verbs))

    # Mean pairwise Jaccard over descriptions: a cheap, model-free proxy for
    # how much the tool surface asks the router to disambiguate.
    descs = [_token_set(s.get("description", "")) for s in schemas]
    sims, n = 0.0, 0
    for i in range(len(descs)):
        for j in range(i + 1, len(descs)):
            union = descs[i] | descs[j]
            if union:
                sims += len(descs[i] & descs[j]) / len(union)
                n += 1
    args = [len((s.get("parameters") or {}).get("properties") or {}) for s in schemas]
    return {
        "environment": module.ENV_ID,
        "tier": module.TIER,
        "n_tools": len(names),
        "tools_sharing_a_verb": shared_verb,
        "mean_description_similarity": round(sims / n, 4) if n else 0.0,
        "mean_args_per_tool": round(sum(args) / len(args), 2) if args else 0.0,
        "max_args_per_tool": max(args) if args else 0,
    }
