"""Deterministic prompt normalization (the `opt_normalize` profile lever).

Strictly rule-based. Using a language model to rewrite the prompt would put a
second model inside a single-model benchmark and make every downstream number
uninterpretable, so the transformations here are a fixed, auditable list:

* spelled-out numbers become digits ("forty percent" -> "40 percent"), because
  arguments must be grounded in a span of the request and a digit is a cleaner
  span than a word;
* "half past two" style clock forms become HH:MM;
* common contractions of negation are expanded, so "don't" and "do not" hit the
  same tokens.

Nothing here adds information the user did not supply.
"""
from __future__ import annotations

import re

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}

_CONTRACTIONS = {
    r"\bdon't\b": "do not", r"\bdoesn't\b": "does not", r"\bdidn't\b": "did not",
    r"\bwon't\b": "will not", r"\bcan't\b": "cannot", r"\bisn't\b": "is not",
    r"\bshouldn't\b": "should not", r"\bwouldn't\b": "would not",
}

_TENS_UNIT = re.compile(
    r"\b(" + "|".join(_TENS) + r")[- ](" + "|".join(_UNITS) + r")\b", re.IGNORECASE)
_TENS_ONLY = re.compile(r"\b(" + "|".join(_TENS) + r")\b", re.IGNORECASE)
_UNITS_ONLY = re.compile(r"\b(" + "|".join(_UNITS) + r")\b", re.IGNORECASE)
_HALF_PAST = re.compile(r"\bhalf past (\d{1,2})\b", re.IGNORECASE)
_QUARTER_PAST = re.compile(r"\bquarter past (\d{1,2})\b", re.IGNORECASE)
_OCLOCK = re.compile(r"\b(\d{1,2}) ?o'?clock\b", re.IGNORECASE)


def normalize(text: str) -> str:
    out = text
    for pattern, replacement in _CONTRACTIONS.items():
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)

    out = _TENS_UNIT.sub(
        lambda m: str(_TENS[m.group(1).lower()] + _UNITS[m.group(2).lower()]), out)
    out = _TENS_ONLY.sub(lambda m: str(_TENS[m.group(1).lower()]), out)
    # Leave "one" and "a"-like small words alone below ten only when they are
    # clearly quantities; a bare "one" is too often a pronoun ("the kitchen one").
    out = _UNITS_ONLY.sub(
        lambda m: str(_UNITS[m.group(1).lower()])
        if _UNITS[m.group(1).lower()] >= 2 else m.group(0), out)

    out = _HALF_PAST.sub(lambda m: f"{int(m.group(1)):02d}:30", out)
    out = _QUARTER_PAST.sub(lambda m: f"{int(m.group(1)):02d}:15", out)
    out = _OCLOCK.sub(lambda m: f"{int(m.group(1)):02d}:00", out)
    return out
