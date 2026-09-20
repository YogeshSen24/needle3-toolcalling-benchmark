"""Embedding helpers for the explicit-prefilter profile and for diagnostics.

Needle's engine is a process-global singleton, so interleaving an embed call
between two agents forces a re-init of roughly a second. Every embedding is
therefore precomputed in one pass at startup — all tool schemas and all case
prompts are known in advance — and looked up at run time.
"""
from __future__ import annotations

import json
import math


def centered_cosine(vectors: list[list[float]]) -> float:
    """Mean pairwise cosine after removing the set mean.

    Needle's embedding space is strongly anisotropic: raw cosine between any
    two tool schemas sits around 0.99, so absolute values carry no information
    and every environment looks equally overlapping. Subtracting the set mean
    restores the spread. Only ranking is meaningful in the raw space.
    """
    if len(vectors) < 2:
        return 0.0
    dim = len(vectors[0])
    mean = [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]
    centered = [[v[i] - mean[i] for i in range(dim)] for v in vectors]
    total, n = 0.0, 0
    for i in range(len(centered)):
        for j in range(i + 1, len(centered)):
            total += cosine(centered[i], centered[j])
            n += 1
    return round(total / n, 4) if n else 0.0


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return num / (na * nb) if na and nb else 0.0


def serialize_tool(schema: dict) -> str:
    return json.dumps(schema, sort_keys=True)


class EmbeddingCache:
    def __init__(self):
        self._vectors: dict[str, list[float]] = {}

    def __contains__(self, text: str) -> bool:
        return text in self._vectors

    def get(self, text: str) -> list[float] | None:
        return self._vectors.get(text)

    def fill(self, texts: list[str]) -> None:
        """Embed everything not already cached, in one engine binding."""
        todo = [t for t in dict.fromkeys(texts) if t not in self._vectors]
        if not todo:
            return
        import needle
        # A one-tool agent: we only want the retrieval head, not a toolset.
        agent = needle.Needle(
            tools=[{"name": "noop", "description": "placeholder",
                    "parameters": {"type": "object", "properties": {}}}],
            system="date: 2026-09-20 Sun 14:00", auto_date=False)
        try:
            for text in todo:
                self._vectors[text] = agent.embed(text)
        finally:
            agent.close()

    def rank(self, query: str, candidates: list[str]) -> list[tuple[float, str]]:
        qv = self._vectors.get(query)
        if qv is None:
            return [(0.0, c) for c in candidates]
        scored = [(cosine(qv, self._vectors[c]), c) for c in candidates
                  if c in self._vectors]
        scored.sort(reverse=True)
        return scored


CACHE = EmbeddingCache()
