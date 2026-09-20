# Needle 3 Tool-Calling Capability Benchmark

What can Cactus Compute's **Needle 3** actually do as a tool-calling model, where does it break,
and how much of the breakage is the harness's fault rather than the model's?

This is **not a model comparison**. No second model is involved. A comparison would only show
that an 8 MB on-device model is worse than a frontier LLM at hard things, which everyone already
assumes. The goal here is the shape of the boundary.

**Read the write-up first:** [Where Needle 3 Works](https://claude.ai/code/artifact/483e29e3-77ba-40f1-bc32-426df201ce02)
— the narrative version, with what the findings mean for building on this model.

| | |
|---|---|
| Full technical report | [reports/needle3_capability_report.md](reports/needle3_capability_report.md) |
| Interactive replay console | [live](https://claude.ai/artifact/E25PPDASPhMFkbFP4MEm4h) · [source](dashboard/index.html) |
| Design rationale and runtime findings | [DESIGN.md](DESIGN.md) |
| First-iteration notes (historical) | [reports/phase1_findings.md](reports/phase1_findings.md) |

## Headline result

Needle 3 base weights, 110 tasks, three environments, 1,910 episodes:

| | baseline harness | optimized harness |
|---|---|---|
| Full-call exact match | 0.364 | **0.482** |
| End-to-end task success | 0.582 | **0.691** |
| Ambient-context tasks | 0.00 | **0.72** |
| Conditional pairs (both branches) | 0 / 5 | 0 / 5 |

Median 591 ms per task, 146 MB peak RAM, CPU only, fully deterministic.
**61% of the model's high-confidence answers (≥ 0.8) were wrong.**

Stacking *every* optimization scored **−1.8 points**. Stacking only the ones that measured
positive scored **+11.8**. That difference is the point of the study.

---

## Install and run

```bash
python -m pip install -r requirements.txt
```

Python 3.11+ (developed on 3.12). The Needle engine and weights download once from Hugging Face
on first use and cache in `~/.cache/cactus-needle/v3/`. No API key is needed for anything here.

```bash
python -m benchmark.runner --experiment all          # everything (~75 min)
python -m benchmark.runner --experiment baseline     # raw Needle, all 3 environments (~2 min)
python -m benchmark.runner --experiment profiles     # the 10 harness configurations
python -m benchmark.runner --experiment toolcount    # catalogue-size sweep
```

Narrow it while iterating:

```bash
python -m benchmark.runner --experiment baseline --environments smart_home --levels 0 1
python -m benchmark.runner --experiment profiles --profiles baseline optimized_selected
```

Ablations and the viewer:

```bash
python -m benchmark.probes.context_rendering              # how to pass ambient context
python -m benchmark.reports.export_viewer results dashboard/data.json
python -m http.server -d dashboard 8000
```

Runs are serial by necessity: Needle's engine is a process-global singleton
(`needle._active[generation]`), so constructing a second agent steals the binding from the
first. Nothing can be parallelised in-process.

## The three environments

Tiered by properties you can count, not by a label:

| tier | environment | tools | tools sharing a leading verb | max args | destructive tools |
|---|---|---|---|---|---|
| T1 | `smart_home` | 10 | 3 | 2 | 0 |
| T2 | `workspace` (calendar + files) | 12 | 3 | 4 | 2 |
| T3 | `business_ops` (CRM, billing, mail) | 20 | **11** | 3 | 3 |

T3 has six tools starting `search_` and five starting `send_`, so the verb carries almost no
routing signal and the object has to. Each environment is a deterministic simulator — no
network, no randomness, no clock reads.

Six difficulty levels per environment: direct → paraphrase → distraction → ambient context →
parallel → dependent/conditional.

## The optimization levers

Each is one field on a `Profile` ([benchmark/profiles.py](benchmark/profiles.py)), run as a
single-factor arm against the identical baseline so every claim has an isolated effect size with
a paired bootstrap CI.

| lever | what it changes | measured effect |
|---|---|---|
| `opt_context_inline` | ambient facts into the user turn | **+11.8 pts** |
| `opt_constraints` | enums, bounds, regex patterns in schemas | **+6.4 pts** |
| `opt_descriptions` | vendor house-style tool descriptions | +2.7 (n.s.) |
| `opt_loop_breaker` | stop when a call repeats | +0.9 (n.s.) |
| `opt_normalize` | deterministic prompt normalization | 0.0 |
| `opt_action_enum` | one tool per device class | −0.9 (n.s.) |
| `opt_triggers` | regex triggers that force a call | −0.9 (n.s.) |
| `opt_prefilter5` | explicit top-5 embedding prefilter | **−10.0 pts** |

## Methodology notes

* **Deterministic scoring.** No LLM judge. End-to-end success compares the *complete* final
  world state, so an extra unwanted action fails a task even when the required one happened.
* **Contamination gate.** Every task is scored against all 192 prompts in the six shipped
  `needle.environments` suites; the run aborts on a match. It caught one on the first run —
  `"Open the blinds."` is verbatim a shipped case, where the vendor labels it a *refusal*.
* **Two scoring contracts.** `raw` scores the model's output; `production` applies the vendor's
  own gates (`validation.ungrounded`, `validation.negation`, confidence threshold). They
  disagree on real cases, so both are always reported.
* **Paired conditionals.** Every "if X then Y" task runs twice with different world state.
  Credit requires both branches. Testing only the positive branch would have scored a model
  that never checks the condition as competent.
* **Ground truth by construction.** Expected end states are produced by executing the gold calls
  against the simulator, so they cannot drift from the world model.

## Layout

```
benchmark/
  adapters/      base.py needle3.py gemini.py registry.py
  environments/  smart_home.py workspace.py business_ops.py
  harness/       loop.py simulator.py context.py toolset.py embedding.py normalize.py
  dataset/       schema.py cases_*.py contamination.py
  evaluator/     contracts.py scoring.py taxonomy.py stats.py
  probes/        context_rendering.py
  runner/        __main__.py
  profiles.py
config/          experiment.yaml models.yaml
dashboard/       index.html          # three.js replay console
results/         raw evidence, committed
```

`adapters/gemini.py` is unused by this study — it survives from an earlier comparison design and
is kept because it is the working reference for adding a second model arm.

## Raw evidence

Committed, not just described: `results/raw_results.jsonl` (every episode, every response
envelope, unfiltered), `results/failures.jsonl` (expected vs actual plus state diff for each
miss), `results/summary.csv`, `results/by_tool_count.csv`, `results/contamination.json`,
`results/environment.json` (hardware, versions, config).

## Limitations

110 tasks is small; per-level cells are six tasks each and should be read as direction, not
magnitude. One author wrote the tasks. There is no held-out split — the winning configuration
was selected on the same data it is reported on, so treat +11.8 as an upper bound. One machine,
CPU only. Single model version, four days after release.

Full list in [the report](reports/needle3_capability_report.md#5-limitations).
