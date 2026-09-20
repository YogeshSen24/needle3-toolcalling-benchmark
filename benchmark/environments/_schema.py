"""Schema builders shared by the three environments.

Every tool is declared once with both a naive and a tuned description, and with
its constraints declared separately from its type. The profile then decides
which text ships and whether the constraints are compiled in. Declaring both
variants side by side is deliberate: it makes the diff between them reviewable
in one place, which matters when the headline claim of chapter 3 is "rewriting
descriptions moved the number by X".
"""
from __future__ import annotations

from typing import Any


def param(profile, type_: str, naive: str, tuned: str, *, enum=None,
          bounds: tuple[int, int] | None = None, pattern: str | None = None) -> dict:
    out: dict[str, Any] = {
        "type": type_,
        "description": tuned if profile.descriptions == "tuned" else naive,
    }
    if profile.constraints == "tight":
        if enum:
            out["enum"] = list(enum)
        if bounds:
            out["minimum"], out["maximum"] = bounds
        if pattern:
            out["pattern"] = pattern
    return out


def tool(profile, name: str, naive: str, tuned: str, properties: dict,
         required: list[str], triggers: list[str] | None = None) -> dict:
    schema = {
        "name": name,
        "description": tuned if profile.descriptions == "tuned" else naive,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }
    # A matching trigger restricts decoding to the matched tools and forces a
    # call, bypassing the confidence threshold — so it is gated on the profile.
    if profile.triggers and triggers:
        schema["triggers"] = list(triggers)
    return schema


def set_path(state: dict, path: str, value: Any) -> None:
    node = state
    keys = path.split(".")
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value


def apply_overrides(state: dict, overrides: dict | None) -> dict:
    for path, value in (overrides or {}).items():
        set_path(state, path, value)
    return state
