# Phase 1 — mechanics verification and first measurements

> **Historical.** This documents the first iteration, which verified the harness against a
> single environment before the study was rebuilt around three tiers and a factorial
> optimization design. The code it describes has been superseded by `benchmark/environments/`
> and `benchmark/dataset/cases_*.py`; the findings carried forward into
> [`needle3_capability_report.md`](needle3_capability_report.md). Kept for provenance.

**Scope.** Needle 3 base weights only. 40 hand-written smart-home cases, levels L0–L6,
5 per level (L6 as 5 paired branches), run under 2 tool granularities × 2 scoring contracts
= 160 scored rows. `cactus-needle==3.0.2`, Intel i5-14400F, 16 GB RAM, Windows 11,
Python 3.12.10, CPU inference, temperature not applicable (deterministic engine).

**Phase 1 is a mechanics check, not a result.** Every cell is n=5, so the bootstrap intervals in
`results/by_complexity.csv` are very wide. Nothing here should be quoted as a headline number.
What follows is what was verified, plus the effects large enough to be worth designing Phase 2
around.

---

## 1. Verification performed

* **All 40 cases audited by hand**, expected vs actual, against the scorer's verdict. No
  disagreements found.
* **Derived metrics cross-checked programmatically**: false-action fired on exactly the four
  L6 negative branches where an action was taken, abstention scored correctly on the injection
  case, pair success correctly required both branches.
* **Two scorer/dataset bugs were found and fixed by this process**, which is what Phase 1 was
  for. Both are recorded in §4.
* **Contamination gate caught a real collision on the first run** before any results existed.

## 2. Measured results (n=5 per cell — indicative only)

Full-call exact match, RAW contract:

| level | verb_split | action_enum | what the level tests |
|---|---|---|---|
| L0 | 1.00 | 0.80 | direct tool match |
| L1 | 1.00 | 1.00 | paraphrased routing |
| L2 | 0.60 | 0.40 | distractors, negation, injection |
| L3 | 0.00 | 0.00 | implicit argument from context |
| L4 | 0.40 | 0.40 | multiple independent calls |
| L5 | 0.00 | 0.20 | dependent chain |
| L6 | 0.00 | 0.00 | conditional decision |

End-to-end task success (final world state) diverges from the above at L0 (action_enum 1.00 vs
0.80) and L6 (0.30 / 0.20 vs 0.00) — see §4 and §3.3 for why that divergence is informative
rather than noise.

Latency per episode: p50 435 ms (verb_split) / 411 ms (action_enum), p95 ≈ 630–680 ms.
Peak RAM 112.6 MB. Mean agent init 1.04 s, excluded from per-query latency.
Conditional pair success: **0/5 in every condition.**

## 3. Effects worth designing Phase 2 around

### 3.1 Arguments are not resolved from system context — by design, not by misconfiguration

Every L3 case failed, and the reasoning strings showed why: Needle resolved the room to
`living_room` regardless of the actual fact, then emitted nothing.

Because a 0% invites the objection "you passed the context wrong", this was tested directly
(`benchmark/probes/context_rendering.py`), comparing every supported system-fact rendering:

| rendering | correct |
|---|---|
| `location: <room>` | 0/5 |
| `user: currently in the <room>` | 0/5 |
| `location:` + `user:` | 0/5 |
| `device: <room> panel` | 0/5 |
| same fact placed in the user text (reference) | **4/5** |

The model can do the resolution; it will not take an argument from the system turn. This is
consistent with the vendor's stated grounding rule — a call carries only values evidenced by the
request — and with system facts being documented for temporal resolution. So the accurate
statement is *"in this benchmark, Needle 3 did not populate tool arguments from ambient system
context under any supported rendering; supplying the same fact in the user text recovered 4/5"*,
not "Needle failed at context".

The 5th in-prompt case is separately interesting: *"The blinds in here are still shut, get them
open"* produced `close_blinds` — a polarity error on "still shut … get them open".

### 3.2 The vendor's production contract discards correct calls

Two cases scored correct under RAW and zero under PRODUCTION. The clearest is L2.04,
*"Whatever you do, don't start any music. Just dim the study to 10 percent."* → Needle produced
the correct `set_light_brightness(study, 10)` at **confidence 1.0**, with
`validation.negation: true`, which the vendor's own harness converts into a refusal.

This is the F6 prediction reproducing systematically. Reporting either contract alone would be
misleading, in opposite directions.

### 3.3 Conditionals: right answer, no reasoning

Pair success was 0/5 everywhere, and the mechanism is visible in the per-case data. On L6.01a
(bedroom at 28°) Needle emitted `set_temperature(bedroom, 22)` without ever reading the
temperature — **the correct end state, so end-to-end success passes**. On L6.01b, same prompt
with the room at 24°, it emitted the identical call, which is now wrong. L6.04 behaves the same
way. Single-branch end-to-end success would have scored this as partial competence; the paired
design shows the condition was never evaluated.

### 3.4 Argument slot bleed across parallel calls

L4 failures share one shape: the correct tools in the correct order, with the room from the
first call leaking into the second. *"Set the bedroom to 20 degrees and put the living room
lights at 60 percent"* → `set_temperature(bedroom, 20)` then
`set_light_brightness(bedroom, 60)`, at confidence 0.96. Three of five L4 cases fail this way.

### 3.5 Confidence does not separate right from wrong

Under RAW/verb_split, 22 of 40 rows are confidence ≥ 0.8 **and** wrong. Confidence 1.0 appears
on correct refusals, on correct calls that the production contract then discards, and on wrong
calls. Calibration is therefore computed against *decision* correctness, and the confidence
signal looks unusable as a correctness proxy on this dataset — which matters directly for the
hybrid-routing experiment in Phase 5.

### 3.6 Tool granularity changes results at every level

verb_split and action_enum differ at L0, L2 and L5, in both directions. Treating granularity as
a measured factor rather than a fixed choice was the right call; a single-shape benchmark would
have reported a materially different ladder.

## 4. Bugs this phase found (and fixed)

1. **Contaminated case.** `"Open the blinds."` is verbatim a shipped `needle.environments`
   prompt — where the vendor labels it an ungrounded *refusal*, the opposite of the label this
   benchmark gave it. Reworded; the gate now reports max similarity 0.538 with nothing flagged.
2. **Lexical confound in L5.** Two dependent-chain prompts used "cooler"/"warm", which collide
   with the `warm white` / `cool white` colour enum; Needle routed to `set_light_color`. Real
   confusion, but it belongs in the similarity experiment, not in the level testing chaining.
   Reworded.
3. **Simulator bug.** `control_lights(action="on", brightness_percent=40)` is a fair reading of
   the schema, but the executor dropped the brightness. A real device API would not. Fixed, and
   L0 action_enum end-to-end rose to 1.00 while full-call exact stayed 0.80 — the intended
   separation between achieving the goal and producing the canonical call.
4. **Double-counted failure statistic.** High-confidence failures were pooled across contracts
   and granularities, reporting the same case up to four times. Now reported per cell.

## 5. Known limitations of this phase

* n=5 per cell. Intervals are wide; no crossover claim is possible and none is made.
* One environment (smart home) and one model arm. The comparison the project exists to make has
  not been run yet.
* L5/L6 currently conflate "did not chain" with "did not need to chain", since a model that
  guesses the right constant passes end-to-end on one branch. The pairing detects this at the
  pair level; per-case metrics still need care when read alone.
* `accept_alternatives` is in the case schema but not yet honoured by the scorer, which makes
  full-call exact match stricter than intended for shapes with several valid encodings (L0.03
  action_enum is the live example). Phase 2 item.
* The contamination gate is lexical only. The embedding-based check described in DESIGN.md §B.6
  is not yet wired in.

## 6. Reproducing

```bash
python -m benchmark.runner --models needle3
python -m benchmark.probes.context_rendering
```

Raw evidence: `results/raw_results.jsonl`, `results/failures.jsonl`,
`results/contamination.json`, `results/environment.json`.
