# Needle 3 Tool-Calling Crossover Benchmark

Measures where Cactus Compute's **Needle 3** is sufficient for tool-calling work and where a
general-purpose LLM becomes necessary. Not a demo — the ladder, the scoring and the fairness
rules are designed to find the boundary, not to argue a side.

**Read the write-up first:** [Where Needle 3 Works](https://claude.ai/code/artifact/483e29e3-77ba-40f1-bc32-426df201ce02)
— the narrative version, with the findings and what they mean for building on this model.

| | |
|---|---|
| Full technical report | [reports/needle3_capability_report.md](reports/needle3_capability_report.md) |
| Interactive replay console | [live](https://claude.ai/artifact/E25PPDASPhMFkbFP4MEm4h) · [source](dashboard/index.html) |
| Design rationale and runtime findings | [DESIGN.md](DESIGN.md) |
| First-iteration notes (historical) | [reports/phase1_findings.md](reports/phase1_findings.md) |

## Headline result

Needle 3, base weights, 110 tasks across three environments:

| | baseline harness | optimized harness |
|---|---|---|
| Full-call exact match | 0.364 | **0.482** |
| End-to-end task success | 0.582 | **0.691** |
| Conditional pairs (both branches) | 0 / 5 | 0 / 5 |
| Ambient-context tasks | 0.00 | **0.72** |

Median 591 ms per task, 146 MB peak RAM, CPU only, fully deterministic.
61% of the model's high-confidence answers (≥ 0.8) were wrong.

---

## Install

```bash
python -m pip install -r requirements.txt
```

Python 3.11+ (developed on 3.12). The Needle engine and weights download once from Hugging Face
on first use and cache in `~/.cache/cactus-needle/v3/`.

## Run it

Needle-only needs no API key and works immediately:

```bash
python -m benchmark.runner --models needle3
```

The full configured experiment:

```bash
python -m benchmark.runner --config config/experiment.yaml
```

A subset of the complexity ladder:

```bash
python -m benchmark.runner --models needle3 --levels 0 1 2 3
```

One scoring contract instead of both:

```bash
python -m benchmark.runner --models needle3 --contract raw
```

## Adding the LLM arms

Set a key — `.env` is the easiest, since a `$env:` variable set in one shell does not reach
another process:

```bash
cp .env.example .env
```

Put the key after `GEMINI_API_KEY=`, then list the models that actually exist rather than
guessing an ID:

```bash
python -m benchmark.runner --list-models
```

Put two of those IDs into `SMALL_MODEL` and `LARGE_MODEL` in `.env`, flip `enabled: true` for
`small_llm` and `large_llm` in `config/models.yaml`, and run all three arms:

```bash
python -m benchmark.runner --models needle3 small_llm large_llm
```

## What comes out

| file | contents |
|---|---|
| `results/raw_results.jsonl` | every turn, every envelope, verbatim |
| `results/summary.csv` | one row per case × model × contract × granularity |
| `results/by_complexity.csv` | accuracy per level with bootstrap CIs |
| `results/overall.csv` | headline rates, latency percentiles |
| `results/conditional_pairs.csv` | L6 pair success (both branches must be right) |
| `results/needle_confidence.csv` | calibration buckets |
| `results/high_confidence_failures.csv` | confidence ≥ 0.8 and wrong |
| `results/failures.jsonl` | every failure with expected, actual and state diff |
| `results/contamination.json` | overlap scores against the vendor's shipped suites |
| `results/environment.json` | hardware, versions, full config |
| `results/fig_*.png` | complexity curve, calibration, accuracy vs latency |

## How fairness is enforced

* **One shared agentic loop** for every model (`harness/loop.py`). Using each vendor's own agent
  runner would confound loop policy with model capability.
* **One canonical context** (`harness/context.py`), rendered per adapter from the same fact set,
  so the LLM cannot quietly receive a better prompt.
* **Two scoring contracts.** `raw` scores the model's output; `production` applies the vendor's
  own gates (`validation.ungrounded`, `validation.negation`, confidence threshold). They disagree
  on real cases, so both are always reported.
* **Contamination gate.** Every prompt is scored against all 192 prompts in the six shipped
  `needle.environments` suites. The run aborts on a match. It caught one on the first run.
* **No Needle-only tuning.** Base weights, no fine-tuning, and `@needle.tool(triggers=...)`
  regexes are excluded from the base comparison because they bypass the confidence threshold.
* **Abstention is scored by action, not prose.** An empty call list, a refusal and a clarifying
  question all count the same, so Needle is not penalised for having no prose and the LLMs are
  not penalised for having some.

## Layout

```
benchmark/
  adapters/   base.py needle3.py gemini.py registry.py
  harness/    loop.py simulator.py context.py toolset.py
  tools/      smart_home.py            # two granularities, one world state
  dataset/    schema.py phase1.py contamination.py
  evaluator/  contracts.py scoring.py taxonomy.py stats.py
  runner/     __main__.py
  reports/    plots.py
  probes/     context_rendering.py     # runtime ablations
config/       experiment.yaml models.yaml
results/
```

## Status

Phase 1 (mechanics, Needle-only, L0–L6, 40 cases) is complete and hand-verified.
Phases 2–6 — full 350–500 case dataset, tool-count and similarity experiments, language
variation, hybrid routing, holdout — are specified in [DESIGN.md](DESIGN.md) §D and not yet built.
