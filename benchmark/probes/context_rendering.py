"""Ablation: how should 'the room the user is in' be given to Needle?

Reporting 0% on implicit-context cases would be unfair if the benchmark simply
picked the wrong supported way to express the fact. This probe compares every
system-fact rendering available, plus an in-prompt upper bound for reference.
"""
import json, needle
from benchmark.environments import smart_home

from benchmark.profiles import BASELINE

TOOLS = smart_home.tools(BASELINE)
CASES = [
    ("Make it a little dimmer in here, around 30%.", "bedroom", "set_light_brightness", {"room":"bedroom","percentage":30}),
    ("Turn the lights off in here.",                 "study",   "turn_off_light",       {"room":"study"}),
    ("The blinds in here are still shut, get them open.","kitchen","open_blinds",       {"room":"kitchen"}),
    ("Put it to 23 degrees in here.",                "living_room","set_temperature",   {"room":"living_room","degrees_c":23}),
    ("Too bright in here, 55 percent please.",       "bedroom", "set_light_brightness", {"room":"bedroom","percentage":55}),
]

RENDERINGS = {
    "location":        lambda r: (f"date: 2026-09-20 Sun 14:00; location: {r}", None),
    "user":            lambda r: (f"date: 2026-09-20 Sun 14:00; user: currently in the {r}", None),
    "location+user":   lambda r: (f"date: 2026-09-20 Sun 14:00; location: {r}; user: currently in the {r}", None),
    "device":          lambda r: (f"date: 2026-09-20 Sun 14:00; device: {r} panel", None),
    "in_prompt(ref)":  lambda r: ("date: 2026-09-20 Sun 14:00", f" (I am in the {r}.)"),
}


def main():
    for label, render in RENDERINGS.items():
        hits = 0; detail = []
        for prompt, room, want_name, want_args in CASES:
            system, suffix = render(room)
            agent = needle.Needle(tools=TOOLS, system=system, auto_date=False)
            agent.reset()
            r = agent.complete(prompt + (suffix or ""))
            calls = r.get("function_calls") or []
            ok = len(calls) == 1 and calls[0]["name"] == want_name and calls[0]["arguments"] == want_args
            hits += ok
            detail.append(("OK " if ok else "   ") + f"{room:11s} -> {json.dumps(calls)[:70]}")
            agent.close()
        print(f"\n{label:16s} {hits}/{len(CASES)}")
        for d in detail: print("   ", d)


if __name__ == "__main__":
    main()
