"""Canonical context, rendered according to the profile.

Ambient facts (today's date, which room the user is in, who they are) are
authored once as data. How they reach the model is an experimental variable,
not an implementation detail: Phase 1 measured 0/5 when a room was supplied as
a system fact and 4/5 when the identical fact was appended to the user turn,
because Needle grounds arguments in spans of the request.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .normalize import normalize

# Needle's recognized system-fact keys, from the official Python docs.
NEEDLE_KEYS = {"date", "locale", "device", "battery", "network", "location",
               "user", "assistant"}


@dataclass(frozen=True)
class Context:
    date: str = "2026-09-20"
    weekday: str = "Sun"
    time: str = "14:00"
    location: str | None = None     # e.g. the room the user is standing in
    user: str | None = None
    facts: tuple[str, ...] = field(default_factory=tuple)

    def fact_set(self) -> set[str]:
        out = {f"date={self.date} {self.weekday} {self.time}"}
        if self.location:
            out.add(f"location={self.location}")
        if self.user:
            out.add(f"user={self.user}")
        out.update(f"fact={f}" for f in self.facts)
        return out

    def render_system(self, include_ambient: bool = True) -> str:
        """Needle's semicolon-delimited fact syntax under recognized keys."""
        parts = [f"date: {self.date} {self.weekday} {self.time}"]
        if include_ambient:
            if self.location:
                parts.append(f"location: {self.location}")
            bits = ([self.user] if self.user else []) + list(self.facts)
            if bits:
                # No dedicated key exists for arbitrary policy facts, and `user`
                # is the only free-text key Needle recognizes.
                parts.append("user: " + "; ".join(bits))
        return "; ".join(parts)

    def inline_suffix(self) -> str:
        """The same ambient facts phrased as part of the request."""
        bits = []
        if self.location:
            bits.append(f"I am in the {self.location}")
        if self.user:
            bits.append(self.user)
        bits.extend(self.facts)
        return f" ({'; '.join(bits)}.)" if bits else ""

    def has_ambient(self) -> bool:
        return bool(self.location or self.user or self.facts)

    @staticmethod
    def from_dict(d: dict | None) -> "Context":
        d = d or {}
        return Context(date=d.get("date", "2026-09-20"), weekday=d.get("weekday", "Sun"),
                       time=d.get("time", "14:00"), location=d.get("location"),
                       user=d.get("user"), facts=tuple(d.get("facts") or ()))

    def to_dict(self) -> dict:
        return {"date": self.date, "weekday": self.weekday, "time": self.time,
                "location": self.location, "user": self.user, "facts": list(self.facts)}


def build_turn(profile, context: Context, prompt: str) -> tuple[str, str]:
    """Return (system_text, user_text) for this profile.

    The date always stays in the system turn: it is what that channel is
    documented to be for. Only the other ambient facts move.
    """
    inline = profile.context_mode == "inline"
    system = context.render_system(include_ambient=not inline)
    user = prompt + (context.inline_suffix() if inline else "")
    if profile.normalize_prompt:
        user = normalize(user)
    return system, user
