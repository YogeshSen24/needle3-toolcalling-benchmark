"""Benchmark runner.

    python -m benchmark.runner --experiment baseline
    python -m benchmark.runner --experiment profiles
    python -m benchmark.runner --experiment toolcount
    python -m benchmark.runner --experiment all

Needle's engine is a process-global singleton, so execution is serial. That is
also what makes the latency numbers meaningful.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from ..adapters.needle3 import Needle3Adapter
from ..dataset import build_all, contamination
from ..environments import describe_tier, load as load_env
from ..evaluator.contracts import RAW, production_at
from ..evaluator.scoring import score_case
from ..harness.context import Context, build_turn
from ..harness.embedding import CACHE, centered_cosine, serialize_tool
from ..harness.loop import run_episode
from ..harness.simulator import Simulator
from ..harness.toolset import build_catalogue
from ..profiles import ALL_PROFILES, BASELINE, OPTIMIZED, get as get_profile

CONTRACTS = [RAW, production_at(0.4)]
TOOL_SIZES = [None, 10, 20, 50, 100]
GOLD_POSITIONS = {"first": 0, "middle": None, "last": 10_000}


# ---------------------------------------------------------------------------

def capture_environment(extra: dict) -> dict:
    import needle
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
    except Exception:
        commit = None
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python": sys.version.split()[0],
        "cactus_needle_version": needle.__version__,
        "needle_generation": 3,
        "weights": "base (no fine-tuning)",
        "git_commit": commit,
        **extra,
    }


def prewarm(cases, profiles) -> None:
    """Embed every tool schema and every prompt once (see harness/embedding)."""
    texts: list[str] = []
    for env_id in {c.environment for c in cases}:
        env = load_env(env_id)
        for profile in profiles:
            texts.extend(serialize_tool(t) for t in env.tools(profile))
    for case in cases:
        texts.append(case.prompt)
    CACHE.fill(texts)


def semantic_overlap(env, profile) -> float:
    """Mean pairwise cosine between tool schemas, using Needle's own retrieval
    head. Token Jaccard is a poor proxy here: short naive descriptions share
    filler words and score as *more* similar than long specific ones."""
    texts = [serialize_tool(t) for t in env.tools(profile)]
    CACHE.fill(texts)
    return centered_cosine([CACHE.get(t) for t in texts])


# ---------------------------------------------------------------------------

def run_cell(adapter, case, profile, *, size=None, gold_position=None,
             similarity="low", seed=20260920, max_steps=4) -> tuple:
    env = load_env(case.environment)
    gold_names = [c["name"] for c in env.translate_gold(
        [x.model_dump() for x in case.expected.calls], profile)]
    context = Context.from_dict(case.context)
    system, user_text = build_turn(profile, context, case.prompt)

    tools = build_catalogue(env, profile, size=size, similarity=similarity,
                            gold_position=gold_position, gold_names=gold_names,
                            seed=seed, prompt=case.prompt)
    sim = Simulator(env, profile, case.state_overrides)
    episode = run_episode(adapter, user_text, tools, system, sim,
                          max_steps=max_steps,
                          loop_breaker=profile.loop_breaker)
    return episode, tools, system, user_text


def emit(rows, raw_f, fail_f, case, episode, tools, system, user_text, profile,
         extra: dict) -> None:
    for contract in CONTRACTS:
        s = score_case(case, episode, contract, profile, tools_offered=tools)
        s.model = "needle3"
        d = s.to_dict()
        rows.append({
            "case_id": case.id, "environment": case.environment, "tier": case.tier,
            "level": case.complexity_level, "level_name": case.level_name,
            "behavior": case.expected.behavior,
            "pair_id": case.properties.pair_id,
            "needs_ambient_context": case.properties.needs_ambient_context,
            "adversarial_type": case.properties.adversarial_type,
            **extra,
            **{k: v for k, v in d.items() if k not in ("state_diff", "gated_out",
                                                       "failure_types")},
            "failure_types": "|".join(d["failure_types"]),
        })
        if not d["full_call_exact"]:
            fail_f.write(json.dumps({
                "case_id": case.id, "environment": case.environment,
                "level": case.complexity_level, "contract": contract.name,
                **extra, "prompt": case.prompt, "sent_text": user_text,
                "system": system,
                "expected": [c.model_dump() for c in case.expected.calls],
                "actual": [c.to_dict() for c in episode.calls],
                "confidence": d["confidence"], "failure_types": d["failure_types"],
                "state_diff": d["state_diff"],
                "reasoning": episode.first.reasoning if episode.first else None,
            }, default=str) + "\n")
    raw_f.write(json.dumps({
        "case_id": case.id, "profile": profile.name, **extra,
        "prompt": case.prompt, "sent_text": user_text, "system": system,
        "n_tools": len(tools), "tool_names": [t["name"] for t in tools],
        "episode": episode.to_dict(),
    }, default=str) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="benchmark.runner")
    ap.add_argument("--experiment", default="baseline",
                    choices=["baseline", "profiles", "toolcount", "all"])
    ap.add_argument("--environments", nargs="*", default=None)
    ap.add_argument("--levels", nargs="*", type=int, default=None)
    ap.add_argument("--profiles", nargs="*", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--max-steps", type=int, default=4)
    args = ap.parse_args(argv)

    cases = build_all(args.environments, args.levels)
    findings = contamination.check(cases)
    csum = contamination.summarize(findings)
    if csum["rejected"]:
        print("ABORT: prompts overlap the vendor's shipped suites:", csum["rejected"])
        return 2

    if args.profiles:
        profiles = [get_profile(p) for p in args.profiles]
    elif args.experiment in ("profiles", "all"):
        profiles = list(ALL_PROFILES)
    else:
        profiles = [BASELINE]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "contamination.json").write_text(
        json.dumps({"summary": csum, "findings": [asdict(f) for f in findings]}, indent=2),
        encoding="utf-8")
    print(f"{len(cases)} cases | contamination max={csum['max_similarity']:.3f} "
          f"mean={csum['mean_similarity']:.3f} rejected={len(csum['rejected'])}")

    print("prewarming embeddings...")
    prewarm(cases, profiles)

    tiers = []
    for env_id in dict.fromkeys(c.environment for c in cases):
        env = load_env(env_id)
        info = describe_tier(env, BASELINE)
        info["embedding_overlap_baseline"] = semantic_overlap(env, BASELINE)
        tiers.append(info)
        print("  tier:", json.dumps(info))

    adapter = Needle3Adapter()
    rows: list[dict] = []
    t0 = time.perf_counter()
    try:
        with (out_root / "raw_results.jsonl").open("w", encoding="utf-8") as raw_f, \
             (out_root / "failures.jsonl").open("w", encoding="utf-8") as fail_f:

            if args.experiment in ("baseline", "profiles", "all"):
                for profile in profiles:
                    start = time.perf_counter()
                    for case in cases:
                        ep, tools, system, text = run_cell(
                            adapter, case, profile, max_steps=args.max_steps)
                        emit(rows, raw_f, fail_f, case, ep, tools, system, text, profile,
                             {"experiment": "profile", "profile": profile.name,
                              "tool_size": len(tools), "gold_position": "natural"})
                    print(f"  {profile.name:<20} {len(cases)} cases "
                          f"in {time.perf_counter() - start:.1f}s")

            if args.experiment in ("toolcount", "all"):
                # Scaling is measured on D0-D2 only: levels that the model can
                # already do, so a drop is attributable to the catalogue rather
                # than to the task.
                sweep = [c for c in cases if c.complexity_level <= 2]
                for size in TOOL_SIZES:
                    for pos_name, pos in GOLD_POSITIONS.items():
                        start = time.perf_counter()
                        for case in sweep:
                            ep, tools, system, text = run_cell(
                                adapter, case, BASELINE, size=size, gold_position=pos,
                                max_steps=args.max_steps)
                            emit(rows, raw_f, fail_f, case, ep, tools, system, text,
                                 BASELINE,
                                 {"experiment": "toolcount", "profile": BASELINE.name,
                                  "tool_size": len(tools), "gold_position": pos_name})
                        print(f"  size={str(size):<5} gold={pos_name:<7} "
                              f"{len(sweep)} cases in {time.perf_counter() - start:.1f}s")
    finally:
        adapter.close()

    (out_root / "environment.json").write_text(json.dumps(capture_environment({
        "experiment": args.experiment,
        "n_cases": len(cases),
        "profiles": [p.name for p in profiles],
        "tiers": tiers,
        "wall_clock_seconds": round(time.perf_counter() - t0, 1),
        "adapter": adapter.describe(),
        "contracts": [c.name for c in CONTRACTS],
    }), indent=2), encoding="utf-8")

    aggregate(rows, out_root)
    print(f"\ndone in {time.perf_counter() - t0:.1f}s | {len(rows)} score rows")
    return 0


# ---------------------------------------------------------------------------

def aggregate(rows: list[dict], out: Path) -> None:
    import pandas as pd
    from ..evaluator import stats

    df = pd.DataFrame(rows)
    if df.empty:
        print("no rows")
        return
    df.to_csv(out / "summary.csv", index=False)

    metrics = ["full_call_exact", "tool_selection_exact", "e2e_success",
               "arg_f1", "decision_correct", "false_action", "missed_action",
               "destructive_action", "gold_tool_offered"]

    prof = df[df.experiment == "profile"]
    if not prof.empty:
        by_level = (prof.groupby(["profile", "contract", "environment", "tier", "level"])
                        [metrics].mean().reset_index())
        ci = []
        for keys, g in prof.groupby(["profile", "contract", "environment", "level"]):
            iv = stats.bootstrap_ci(g["full_call_exact"].tolist())
            ci.append(dict(zip(["profile", "contract", "environment", "level"], keys),
                           ci_low=iv.low, ci_high=iv.high, n=iv.n))
        by_level = by_level.merge(pd.DataFrame(ci),
                                  on=["profile", "contract", "environment", "level"],
                                  how="left")
        by_level.to_csv(out / "by_level.csv", index=False)

        (prof.groupby(["profile", "contract", "environment"])[metrics].mean()
             .reset_index().to_csv(out / "by_environment.csv", index=False))

        overall = prof.groupby(["profile", "contract"])[metrics].mean().reset_index()
        lat = prof.groupby(["profile", "contract"])["latency_ms"]
        overall["latency_p50"] = lat.median().values
        overall["latency_p95"] = lat.quantile(0.95).values
        overall.to_csv(out / "by_profile.csv", index=False)

        base = prof[(prof.profile == "baseline") & (prof.contract == "raw")]
        base_map = dict(zip(base.case_id, base.full_call_exact))
        deltas = []
        for name, g in prof[prof.contract == "raw"].groupby("profile"):
            m = dict(zip(g.case_id, g.full_call_exact))
            iv = stats.paired_diff_ci(m, base_map)
            deltas.append({"profile": name, "delta_vs_baseline": round(iv.mean, 4),
                           "ci_low": round(iv.low, 4), "ci_high": round(iv.high, 4),
                           "n_shared": iv.n, "significant": stats.excludes_zero(iv)})
        pd.DataFrame(deltas).sort_values("delta_vs_baseline", ascending=False) \
            .to_csv(out / "profile_deltas.csv", index=False)

        pairs = prof[prof.pair_id.notna()]
        if not pairs.empty:
            pk = pairs.groupby(["profile", "contract", "pair_id"])["full_call_exact"] \
                      .all().reset_index()
            pk.groupby(["profile", "contract"])["full_call_exact"].mean().reset_index() \
              .rename(columns={"full_call_exact": "pair_success"}) \
              .to_csv(out / "conditional_pairs.csv", index=False)

        needle = prof[prof.confidence.notna()]
        buckets = (needle.assign(bucket=(needle.confidence * 10).clip(0, 9).astype(int) / 10)
                         .groupby(["profile", "contract", "bucket"])
                         .agg(n=("case_id", "size"),
                              decision_correct=("decision_correct", "mean"),
                              full_call_exact=("full_call_exact", "mean")).reset_index())
        buckets.to_csv(out / "needle_confidence.csv", index=False)
        needle[(needle.confidence >= 0.8) & (~needle.full_call_exact)] \
            .to_csv(out / "high_confidence_failures.csv", index=False)

    tc = df[df.experiment == "toolcount"]
    if not tc.empty:
        (tc.groupby(["contract", "tool_size", "gold_position"])
           [["full_call_exact", "gold_tool_offered", "e2e_success", "latency_ms"]]
           .mean().reset_index().to_csv(out / "by_tool_count.csv", index=False))

    print("\n--- full-call exact match, RAW contract, by environment x level ---")
    raw = df[(df.contract == "raw") & (df.experiment == "profile")]
    if not raw.empty:
        print(raw.pivot_table(index="level", columns=["environment", "profile"],
                              values="full_call_exact").round(2).to_string())


if __name__ == "__main__":
    raise SystemExit(main())
