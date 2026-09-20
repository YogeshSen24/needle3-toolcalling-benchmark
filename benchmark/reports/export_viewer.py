"""Export a compact JSON bundle for the web viewer.

The viewer's job is to make the benchmark inspectable rather than decorative:
for any case you can see the exact prompt sent, the exact tool calls Needle
produced, what the world looked like before and after, and what should have
happened. Failures are the interesting part, so nothing is filtered out.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..dataset import build_all
from ..environments import describe_tier, load as load_env
from ..profiles import BASELINE, OPTIMIZED


def build_bundle(results_dir: Path, profiles: tuple[str, ...] = ("baseline", "optimized_all")) -> dict:
    cases = {c.id: c for c in build_all()}

    episodes: dict[str, dict] = {}
    raw_path = results_dir / "raw_results.jsonl"
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("experiment") != "profile" or rec.get("profile") not in profiles:
            continue
        episodes.setdefault(rec["case_id"], {})[rec["profile"]] = {
            "sent_text": rec["sent_text"],
            "system": rec["system"],
            "n_tools": rec["n_tools"],
            "calls": rec["episode"]["calls"],
            "steps": rec["episode"]["steps"],
            "confidence": (rec["episode"]["turns"][0]["response"]["confidence"]
                           if rec["episode"]["turns"] else None),
            "reasoning": (rec["episode"]["turns"][0]["response"]["reasoning"]
                          if rec["episode"]["turns"] else None),
            "suppressed": (rec["episode"]["turns"][0]["response"]["suppressed_calls"]
                           if rec["episode"]["turns"] else []),
            "validation": (rec["episode"]["turns"][0]["response"]["validation"]
                           if rec["episode"]["turns"] else {}),
            "final_state": rec["episode"]["final_state"],
        }

    scores: dict[str, dict] = {}
    summary = results_dir / "summary.csv"
    if summary.exists():
        import pandas as pd
        df = pd.read_csv(summary)
        df = df[(df.contract == "raw") & (df.experiment == "profile")
                & df.profile.isin(profiles)]
        for _, row in df.iterrows():
            scores.setdefault(row["case_id"], {})[row["profile"]] = {
                "full_call_exact": bool(row["full_call_exact"]),
                "e2e_success": bool(row["e2e_success"]),
                "false_action": bool(row["false_action"]),
                "failure_types": (str(row["failure_types"]).split("|")
                                  if isinstance(row["failure_types"], str)
                                  and row["failure_types"] else []),
            }

    envs = {}
    for env_id in ("smart_home", "workspace", "business_ops"):
        env = load_env(env_id)
        envs[env_id] = {
            **describe_tier(env, BASELINE),
            "initial_state": env.initial_state(),
            "tools": [{"name": t["name"], "description": t["description"],
                       "params": list((t.get("parameters") or {}).get("properties", {}))}
                      for t in env.tools(BASELINE)],
            "tools_tuned": [{"name": t["name"], "description": t["description"]}
                            for t in env.tools(OPTIMIZED)],
        }

    out_cases = []
    for cid, case in cases.items():
        if cid not in episodes:
            continue
        out_cases.append({
            "id": cid,
            "environment": case.environment,
            "tier": case.tier,
            "level": case.complexity_level,
            "level_name": case.level_name,
            "prompt": case.prompt,
            "context": case.context,
            "state_overrides": case.state_overrides,
            "expected": {
                "behavior": case.expected.behavior,
                "calls": [c.model_dump() for c in case.expected.calls],
                "final_state": case.expected.final_state,
            },
            "properties": case.properties.model_dump(),
            "runs": episodes[cid],
            "scores": scores.get(cid, {}),
        })
    out_cases.sort(key=lambda c: (c["tier"], c["level"], c["id"]))

    env_meta = {}
    env_json = results_dir / "environment.json"
    if env_json.exists():
        env_meta = json.loads(env_json.read_text(encoding="utf-8"))

    return {"environments": envs, "cases": out_cases, "profiles": list(profiles),
            "meta": env_meta}


def main(results_dir="results/main", out="dashboard/data.json") -> None:
    bundle = build_bundle(Path(results_dir))
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(bundle, separators=(",", ":"), default=str),
                      encoding="utf-8")
    kb = target.stat().st_size / 1024
    print(f"wrote {target} ({kb:.0f} KB, {len(bundle['cases'])} cases)")


if __name__ == "__main__":
    import sys
    main(*(sys.argv[1:] or []))
