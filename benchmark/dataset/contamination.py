"""Contamination gate against the vendor's own shipped test suites.

DESIGN.md F9: `cactus-needle` ships six environments with frozen acceptance
suites, and several of their prompts are near-duplicates of the examples in the
original benchmark spec. Benchmarking a model on prompts shipped alongside it
would not measure generalisation.

Lexical overlap runs always (no model needed). Embedding similarity is optional
because it requires binding the engine.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def shipped_prompts() -> list[tuple[str, str]]:
    """Every prompt in every shipped needle.environments acceptance suite."""
    out: list[tuple[str, str]] = []
    try:
        from needle import environments
    except Exception:
        return out
    for name in getattr(environments, "_NAMES", ()):
        try:
            module = __import__(f"needle.environments.{name}", fromlist=["TEST_CASES"])
        except Exception:
            continue
        for case in getattr(module, "TEST_CASES", []):
            query = case.get("query")
            if query:
                out.append((name, query))
    return out


@dataclass
class Finding:
    case_id: str
    prompt: str
    score: float
    nearest_env: str
    nearest_prompt: str
    verdict: str   # "clean" | "review" | "reject"


def check(cases, review_at: float = 0.60, reject_at: float = 0.85) -> list[Finding]:
    """Score every case against the shipped corpus by token Jaccard.

    Thresholds are preregistered here rather than tuned after seeing results.
    """
    corpus = shipped_prompts()
    findings: list[Finding] = []
    for case in cases:
        prompt = case.prompt if hasattr(case, "prompt") else case["prompt"]
        cid = case.id if hasattr(case, "id") else case["id"]
        best, best_env, best_prompt = 0.0, "", ""
        for env, shipped in corpus:
            score = jaccard(prompt, shipped)
            if score > best:
                best, best_env, best_prompt = score, env, shipped
            if normalize(prompt) == normalize(shipped):
                best, best_env, best_prompt = 1.0, env, shipped
                break
        verdict = "clean"
        if best >= reject_at:
            verdict = "reject"
        elif best >= review_at:
            verdict = "review"
        findings.append(Finding(cid, prompt, round(best, 3), best_env, best_prompt, verdict))
    return findings


def summarize(findings: list[Finding]) -> dict:
    return {
        "n_cases": len(findings),
        "n_shipped_prompts": len(shipped_prompts()),
        "rejected": [f.case_id for f in findings if f.verdict == "reject"],
        "review": [f.case_id for f in findings if f.verdict == "review"],
        "max_similarity": max((f.score for f in findings), default=0.0),
        "mean_similarity": (
            round(sum(f.score for f in findings) / len(findings), 4) if findings else 0.0
        ),
    }
