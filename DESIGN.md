# Needle 3 Tool-Calling Crossover Benchmark — Design Proposal

Status: **proposal, pre-implementation.** Nothing in §A–D is built yet.
Date: 2026-09-20. Written after live inspection of `cactus-needle==3.0.2`.

---

## 0. What was verified against the real runtime (not documentation)

Everything below was observed by running `cactus-needle==3.0.2` on this machine
(Intel i5-14400F, 16 logical cores, 15.8 GB RAM, Windows 11, Python 3.12.10, CPU inference).
Probe scripts will be promoted to `benchmark/probes/` so every finding is reproducible.
These findings change the design, so they come first.

### 0.1 API surface (read from installed source, `needle/__init__.py`)

```python
needle.Needle(tools=None, system=None, weights=None, tool_index_path=None,
              buffer_size=65536, auto_date=True, generation=None)   # generation defaults to 3
agent.complete(text, max_new_tokens=512) -> dict
agent.run(query, max_steps=8, max_new_tokens=512, strict=True) -> dict   # adds "results"
agent.embed(text) -> list[float]      # 3072-dim, Needle 3 only
agent.reset(); agent.close()
needle.extract(text, schema, ...)
```

Observed response envelope (real output, not copied from docs):

```json
{"type":"call","success":true,"error":null,"error_code":null,"reason":null,
 "function_calls":[{"name":"turn_off_light","arguments":{"room":"kitchen"}}],
 "suppressed_calls":[],
 "reasoning":"Query 'turn off the kitchen light' -> room 'kitchen', action 'turn off'.",
 "confidence":0.9923,"prefill_tps":609.7,"decode_tps":295.4,"peak_ram_mb":99.7,
 "validation":{"ungrounded":[],"negation":false}}
```

### 0.2 Findings that alter the experiment

**F1 — There is no `escalate` field.** The spec assumed one. `grep -rn escalate` over the
installed package returns nothing. Escalation must be *derived* by the harness, not read.
Two fields the spec did not anticipate do exist and matter: `reason`, and `validation`
(`{ungrounded: [paths], negation: bool}`).

**F2 — Needle 3 is deterministic.** The same query run 5× at 20 tools produced byte-identical
calls *and* identical confidence (0.9937). Consequence: **1 run per case for Needle**, repeats
(3–5) only for the API models. This cuts the compute budget substantially. It is treated as a
measured property, not an assumption — a determinism check runs as a standing assertion.

**F3 — The C engine is a process-global singleton.** `_active[generation] = self`; constructing a
second `Needle` re-runs `needle_init` and steals the binding. **In-process parallelism is unsafe.**
The runner must use a process pool with one agent per process, or run Needle serially. Latency
numbers are only trustworthy at a controlled concurrency of 1.

**F4 — A large tool catalogue causes silent, high-confidence misses.** Same query, same gold
tool, varying catalogue size (distant-domain fillers):

| tools | 1 | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|
| correct call | yes | yes | yes | yes | **no (empty)** | **no (empty)** |
| confidence | 0.994 | 0.994 | 0.996 | 0.994 | **0.962** | **1.000** |

At n=100 the model returned zero calls with **confidence 1.0**. Confidence does not flag this
failure. This is the most decision-relevant result found so far.

**F5 — The n=50 failure is *not* a retrieval-ranking failure, and it is position-sensitive.**
Ranking all 100 serialized schemas against the query with `agent.embed()`, the gold tool ranks
**#1 of 100** — the retrieval head has the right answer. Yet at n=50 the outcome depends on where
the gold tool sits in the list: position 49 (last) → correct call; positions 0, 10, 25 → empty.
And with 49 *semantically near* smart-home fillers it succeeded (conf 0.717) where 49 *distant*
fillers made it fail. Consequences: **gold-tool position is a controlled factor**, not an
incidental detail; and the "semantic overlap hurts Needle" hypothesis is not safe to assume in
either direction. Recall@5 from `embed()` is logged separately so retrieval failure and
selection failure can be told apart.

**F6 — Raw output and the vendor's own production contract disagree.** The shipped harness
(`needle/environments/_harness.py`) discards any call when `validation.ungrounded` or
`validation.negation` is set, and applies a confidence gate ("act on a call only at or above the
threshold, otherwise treat it as a refusal"). Observed case: *"don't turn the bedroom light off,
just dim it to 20%"* → Needle produced the **correct** `set_light_brightness(bedroom, 20)` at
confidence 0.875, but with `validation.negation: true`, which the vendor contract converts into a
refusal. **Every metric is therefore reported under two contracts** (§C.1).

**F7 — A real Level-4 failure appeared in the first six queries.**
*"turn off the bedroom lights and set the living room to 22 degrees"* →
`set_temperature(room="bedroom", degrees_c=22)` at **confidence 0.990**, while its own
`reasoning` string said `room 'living_room'`. Argument-slot bleed across parallel calls, high
confidence, wrong. The benchmark will find signal; difficulty does not need manufacturing.

**F8 — A real Level-6 failure, identical across both branches.** `run()` on *"if the bedroom is
warmer than 26 degrees set it to 22, otherwise leave it alone"* never called `get_temperature`;
it emitted `set_temperature(bedroom, 26)` — lifting 26 out of the condition text — and produced
**the same call whether the mocked room was 28° or 24°**. The paired-case design (§B.4) detects
this as condition-insensitivity rather than merely "wrong answer".

**F9 — Contamination risk is concrete.** The package ships six environments with frozen test
suites whose prompts closely resemble examples in the spec — e.g. shipped
`"don't turn on the study lights"` against the spec's negation case, and shipped
`"turn off the bedroom lights and set the thermostat to 18 degrees"` against the spec's Level-4
example. A contamination gate (§B.6) is mandatory, not optional.

**F10 — The vendor's tool idiom differs from the spec's.** The spec proposes verb-split tools
(`turn_on_light`, `turn_off_light`, `set_light_brightness`). The shipped environment uses one
tool per device class with an action enum
(`control_lights(room, action, brightness_percent, color)`), and the tool-design guide warns that
above five tools "an unselected tool is unreachable". Picking one shape silently decides part of
the answer, so **tool granularity becomes an explicit factor** (§B.5), measured on both shapes.

**F11 — No LLM provider key is configured** on this machine (only an unrelated `RENDER_API_KEY`).
The Needle arm can run today; the two API arms are blocked on a key (§E).

---

## A. Proposed architecture

```
benchmark/
  adapters/
    base.py          # ModelAdapter ABC -> normalized ModelResponse; no scoring logic here
    needle3.py       # cactus-needle; one agent per process (F3); full envelope capture
    openai_compat.py # shared OpenAI-style native tool-calling client (OpenRouter/OpenAI/...)
    small_llm.py     # thin config binding over openai_compat
    large_llm.py     # thin config binding over openai_compat
    registry.py
  harness/
    loop.py          # THE shared agentic loop - identical for every model
    simulator.py     # deterministic world state; no network
    context.py       # canonical context object -> per-adapter rendering
    toolset.py       # catalogue assembly: granularity, size, similarity, gold position
  tools/
    smart_home.py travel.py calendar.py commerce.py files.py confusable.py synthetic.py
  dataset/
    schema.py generator.py contamination.py
    cases.dev.jsonl cases.holdout.jsonl knowledge.jsonl
  evaluator/
    contracts.py     # raw vs production scoring contracts (F6)
    scoring.py metrics.py taxonomy.py calibration.py stats.py judge.py
  runner/
    benchmark.py pool.py   # process pool; concurrency 1 for Needle (F3)
  reports/ dashboard/ config/ results/ probes/
```

Three design commitments, each forced by a finding above:

1. **One shared loop, not each vendor's own loop.** `harness/loop.py` drives every model
   identically: get calls → execute against `simulator.py` → feed results back in that model's
   native turn format → repeat to `max_steps`. For Needle this reproduces `run()`'s semantics
   (`complete(json.dumps(results))`) but with our instrumentation and *without* baking in
   `strict=` grounding, which is a scoring contract rather than a harness behaviour. Using each
   vendor's own agent loop would confound loop policy with model capability.

2. **One canonical context, rendered per adapter.** A `Context` object (date, location, current
   room, user preferences, working hours) is the single source of truth. Needle receives it as
   system facts in its recognized-key format (`date:`, `location:`, `user:` …); the LLMs receive
   the *same facts* in a system message with no added guidance, examples or instructions. A test
   asserts the two fact sets are equal, so "the LLM quietly got a better prompt" cannot creep in.

3. **Needle gets its official mechanisms, and we measure what they cost.** Above five tools,
   retrieval engages — that is the supported path, so we use it rather than hacking around it,
   and log recall@5 via `embed()` (F5) alongside. **`@needle.tool(triggers=[...])` regexes are
   excluded from the base comparison** — they bypass the confidence threshold and amount to
   hand-tuning one model only — and are reserved for a clearly-labelled ablation.

---

## B. Dataset schema

### B.1 Case record

```jsonc
{
  "id": "smart_home.L2.017",
  "split": "dev",                      // dev | holdout   (70/30, hashed on id, frozen)
  "environment": "smart_home",
  "complexity_level": 2,
  "prompt": "Don't turn it off, just dim the bedroom lights to 20%.",

  "context": {"date":"2026-09-20","time":"14:00","current_room":"bedroom",
              "location":"home","user_prefs":{},"facts":[]},

  "toolset": {                         // how the catalogue is built, not a literal tool list
    "base": "smart_home",
    "granularity": "verb_split",       // verb_split | action_enum      (F10)
    "size": 10,                        // 5 | 10 | 20 | 50 | 100        (F4)
    "similarity": "low",               // low | medium | high
    "gold_positions": [0, 4, 9],       // evaluated at each position    (F5)
    "filler_seed": 20260920
  },

  "expected": {
    "behavior": "tool_call",           // tool_call | multiple_tool_calls | no_call | clarify
    "calls": [{"name":"set_light_brightness",
               "arguments":{"room":"bedroom","percentage":20}}],
    "order_matters": false,
    "arg_tolerance": {"percentage":{"type":"int_exact"}},
    "accept_alternatives": [],         // other genuinely correct call sequences
    "final_state": {"lights.bedroom.brightness":20,"lights.bedroom.on":true}
  },

  "properties": {
    "negation": true, "multi_tool": false, "ambiguous": false,
    "requires_external_knowledge": false, "conditional": false,
    "language_variant": "canonical",   // canonical|natural|colloquial|verbose|typo|indirect
    "adversarial_type": null,          // negation|correction|irrelevant|multi_number|
                                       // injection|contradiction|missing_arg|unsupported
    "pair_id": null,                   // links the two branches of an L6 conditional  (F8)
    "source_case": null                // links a language variant to its canonical case
  },

  "provenance": {"author":"generator","reviewed_by":null,
                 "contamination_checked": true, "max_shipped_similarity": 0.61}
}
```

### B.2 Behaviour types — and the honest handling of "escalation" (F1)

Needle exposes no `escalate` signal and does not emit prose, so the gradable unit is the
**action decision**, identical for every model:

| expected | scored correct when |
|---|---|
| `tool_call` / `multiple_tool_calls` | the required calls are produced |
| `no_call` | **no action taken** — empty `function_calls`, *or* a clarifying question, *or* a refusal |
| `clarify` | same as `no_call`; asking a question earns no extra credit and costs nothing |

Whether a model *additionally* produced clarifying text is recorded as a descriptive column
(`asked_clarification`) and never folded into accuracy. This satisfies both fairness rules at
once: Needle is not penalised for producing no prose, and the LLMs are not penalised for
producing some. `suppressed_calls` is recorded but does **not** count as an action under RAW.

### B.3 Size and splits

350–500 primary cases as specified, plus the Level-10 knowledge set kept **entirely separate**
from every headline number. Split 70/30 dev/holdout by hash of `id`, frozen at generation time.
Holdout is executed **once**, at the end. Language variants are forced into the *same* split as
their canonical parent so variants cannot leak across the boundary.

### B.4 Conditional pairs (L6)

Every L6 case is generated as a `pair_id`-linked pair: identical prompt, differing simulator
state (e.g. 28° vs 24°). Credit requires **both** branches correct, which is what separates
reasoning from a lucky constant — F8 showed the same call emitted for both.

### B.5 Factors crossed over the ladder

`granularity` × `size` × `similarity` × `gold_position`, applied to a Level 0–5 subset rather
than the whole ladder (full crossing is combinatorially wasteful). Each factor is recorded on
every run record so the effects are separable in analysis.

### B.6 Contamination gate (F9)

Before a case enters the dataset it is checked against all `TEST_CASES` in the six shipped
`needle.environments` modules and the published blog examples: normalized exact match →
rejected; cosine similarity above a preregistered threshold (via `agent.embed()`) → flagged for
human review. `max_shipped_similarity` is stored on every case and its distribution is published
in the report.

---

## C. Scoring system

### C.1 Two contracts, always reported side by side (F6)

* **RAW** — score `function_calls` exactly as returned.
* **PRODUCTION** — the vendor's own contract: drop calls when `validation.ungrounded` or
  `validation.negation` is set, and apply a confidence gate (τ swept; default 0.4). For the API
  models the analogous contract is "act only on returned tool calls", so the arms stay
  comparable.

Reporting only RAW would flatter Needle on cases its own guardrails would block. Reporting only
PRODUCTION would punish it for correct answers its guardrails discard. Both are published.

### C.2 Metrics (deterministic and programmatic)

Tool-selection accuracy · argument exact match · argument F1 · **full-call exact match** (the
headline) · sequence success (dependency order) · **end-to-end task success** (final simulator
state equality — the metric the report leads with) · false-action rate · missed-action rate ·
unsupported-request rejection · **retrieval recall@5** (Needle diagnostic, F5) · latency
p50/p90/p95/p99 · `peak_ram_mb` · `prefill_tps`/`decode_tps` · tokens and API cost (Needle:
`api_cost = 0` *plus* separately measured wall-clock and RAM — never reported as "free") ·
determinism and variance across repeats.

An LLM judge is used **only** for the Level-10 open-ended set and for optional failure-cause
hypotheses, which are written to disk labelled `hypothesized_failure_reason` and excluded from
every score.

### C.3 Calibration

Confidence is bucketed in tenths against **decision-correctness**, not call-correctness —
necessary because a correct refusal returned `confidence: 1.0`. The high-confidence-failure
report (`confidence ≥ 0.8` ∧ incorrect) is generated verbatim from raw records; F4 and F7
already guarantee it will not be empty.

### C.4 Statistics

Bootstrap confidence intervals (10k resamples) on every reported rate. Crossover is declared
only on the preregistered rule: **a gap whose 95% bootstrap CI excludes zero across at least two
adjacent complexity levels**, using paired resampling over shared cases. Single-level gaps are
reported as observations, never as crossovers.

---

## D. Implementation plan

**Phase 1 — mechanics, then stop for inspection** (as specified): repo and config; `ModelAdapter`
ABC and normalized response; Needle adapter with full envelope capture and process isolation
(F3); OpenAI-compatible adapter; smart-home tools in **both** granularities; simulator; shared
loop; ~5 hand-written cases per level for L0–L6 with the contamination gate applied;
deterministic evaluator under both contracts; a Needle-only run; first plots; **manual
verification of every score by hand.**

**Phase 2** — full dataset generation to 350–500 cases across all environments; holdout frozen.
**Phase 3** — tool-count, similarity, granularity and position experiments.
**Phase 4** — language-variation and adversarial suites.
**Phase 5** — hybrid routing threshold sweep; confidence-failure analysis; calibration.
**Phase 6** — holdout executed once; report, `blog_findings.json`, dashboard.

Entry points exactly as specified (`python -m benchmark.runner --config …`, `--models`,
`--levels`), plus `--contract raw|production|both` and `--splits dev|holdout`.
`results/environment.json` captures CPU/GPU/RAM/OS/Python version/`cactus-needle` version/engine
version/model IDs/temperature/git commit/timestamp.

---

## E. Decisions needed before Phase 1

1. **LLM provider and the two model IDs** — no key is configured (F11). Needle-only Phase 1 can
   start immediately; the comparison arms cannot.
2. **Tool granularity** — measure both shapes as a factor (recommended), or fix one?
3. **Two-contract scoring** — confirm. It roughly doubles the result tables, but F6 shows a
   single contract would misreport.
4. **Cost ceiling** for API calls, which drives repeats × cases × tool-count conditions.
