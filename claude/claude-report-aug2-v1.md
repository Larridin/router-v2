# Model Router Project — Plain-Language Report

**Date:** 2026-08-02 · **Version:** v1
**Project home:** `~/dev/exp/router-v2/claude/` (routerlab)

## The idea in one paragraph

A "model router" answers one question: *for this prompt, which AI model should
handle it — the expensive smart one or the cheap fast one?* We studied how
Weave's production router does this, rebuilt its core brain ourselves, tested
whether we could beat it, and ended up with our own working router trained on
the models we actually care about (fable, opus, sonnet, haiku, deepseek, kimi).

## The story so far, step by step

**1. We learned how the pros do it.** Weave's router turns each prompt into a
small numeric fingerprint (an "embedding"), finds which group of similar
prompts it belongs to, and looks up a pre-computed table: "in this group,
model X scores this well and costs this much." Pick the best trade-off. All
the intelligence is baked into tables ahead of time; routing itself takes
milliseconds.

**2. We built a report card system (the evals).** To judge any router, you
need an answer key: a big spreadsheet where every row is a prompt and every
column is a model, filled with "did it get it right?" and "what did it cost?".
We got one for free from public research data (RouterBench, 36,000 prompts).
Then judging a router is simple: let it pick, look up what its pick earned.

**3. We rebuilt Weave's algorithm and tried to beat it.** Four serious
attempts. Most failed in an instructive way — fancier prediction math didn't
help, a cheap LLM judging prompts was too slow (~1 second per request),
collapsing models into "low/med/high" lost accuracy. **One thing genuinely
worked**: changing the *decision* from "pick the highest blended score" to
"pick the best value in actual dollars — predicted success minus λ × predicted
cost." That one change delivered ~98% of the flagship model's quality at
**half the cost**.

**4. We made it real for our models.** We ran 1,290 test questions (math,
code, knowledge) through our 7 models — 9,030 API calls, $33 total — building
our own answer key. Then we trained the router on it. Headline: a router over
these 7 models gets ~90% of opus-5's quality at **one-third of its price**,
and the data says near-opus quality is achievable at 1/8th the price with more
training data.

**5. We sanity-checked against Codex's parallel attempt.** It reached the
same conclusion by a different road: the core algorithm is basically settled;
data and the decision rule are where wins live.

## How to use what's built

Everything lives in `~/dev/exp/router-v2/claude/`. The four things you'd
actually touch:

**Route a prompt** (the working router, trained on public data):

```bash
uv run python -m routerlab.bundle route "your prompt here" --lam 30
```

`--lam` is the thrift dial: `0` = always pick the predicted-best model, higher
= save money more aggressively. It returns the chosen model, predicted success
rate, estimated cost, and takes ~80ms.

**Add a new model when one ships** (say gpt-5.7):

```bash
uv run python -m routerlab.labelmatrix run openai/gpt-5.7   # ~$5, runs the test questions
uv run python -m routerlab.labelmatrix report               # grade it
uv run python -m routerlab.roster                           # retrain + fresh comparison
```

The corpus and every other model's results are cached forever — you only ever
pay for the new column.

**Read the results** — human-readable reports in `results/`:

- `relative_gains.md` — the headline comparison vs Weave's algorithm
- `roster_eval.md` — our 7 current models
- `hypotheses.md` — what we tried and what won
- `tiers.md` — low/med/high abstraction analysis
- frontier charts as PNGs

**Re-run any experiment**: `uv run pytest` (22 tests), then
`uv run python -m routerlab.run`, `.hypotheses`, `.tiers`, `.roster` —
everything reproduces from cached data at zero API cost.

## The three lessons, if you remember nothing else

1. **Data beats cleverness.** Every algorithm improvement we tried was worth
   less than simply having more graded examples. Budget goes to the answer
   key, not the math.
2. **Decide in dollars, not scores.** The single biggest win was making the
   router reason about actual predicted cost per request.
3. **Keep the routing path dumb and fast** (~10ms embedding + table lookup).
   Anything smart and slow — LLM judges, session analysis — belongs offline or
   off the hot path, feeding the tables.

## What's deliberately not built yet

- The production wrapper: session stickiness, prompt-cache economics,
  fallbacks — the "other 80%" of a production router (see the Weave repo for
  the reference design).
- An agentic/tool-use test set — the biggest gap, since our real traffic is
  agentic. Both are known roads, not mysteries.

## Addendum (later Aug 2): the quality dial — how to handle the quality↔cost scale

Cross-checked our router on Codex's independent benchmark (LLMRouterBench:
newer models, harder tasks, SWE-bench). What held and what we learned about
running quality-consciously:

**Cross-benchmark verdict.** Our dollars-based rule wins the cheap-to-mid band
there too (+1.1 to +2.3 pts at matched cost, significant) and fixes agentic
routing (SWE-bench 0.282 vs Weave's 0.197 at equal spend). Weave's simpler
algorithm keeps the pure-quality crown on that data (our averaged estimator is
too smooth at the top end). So: our router for the savings regime, estimator
work still owed at the top.

**How to expose the dial (0 = max savings, 100 = max quality):**

1. **Never map the slider linearly to λ** — λ is log-scaled and most of its
   range does nothing. Sweep λ densely, find every point where the *served
   model mix* changes (314 on the cross-bench data), and space the slider
   across those breakpoints so every notch visibly does something.
2. **The top is flat: default ≈ 70–80, not 100.** From dial 100 down to ~70,
   delivered quality is statistically unchanged while cost falls ~33% — even
   at "max quality" the router sends only ~19% of traffic to the priciest
   model, because per-prompt the priciest model often isn't the best one.
3. **Floor the dial at ~20 — never expose the cliff.** Below ~10 the mix
   collapses onto one cheap model and quality craters (0.55 → 0.41). Same
   idea as Weave's production AlphaFloor.
4. **Watch downgrade regret, not average quality.** At matched spend we route
   ~40% of prompts to a ≥2×-cheaper model; ~11% of those downgrades lose a win
   the pricier route had. Users feel that 11% (a visible wrong answer) far more
   than they notice the invisible savings — perceived quality is asymmetric.
5. **Fix the asymmetry with a confidence floor on downgrades**: only pick the
   cheaper model when its calibrated P(success) is high (≈0.7) or within ε of
   the best model's. Our one observed bad downgrade was taken at P̂=0.59; the
   good ones at ≥0.75 — the floor separates them cleanly. (Only expressible
   because our estimates are calibrated probabilities, not raw scores.)
   Stronger form: add a regret term, score = P̂ − λ·cost − ρ·(P̂_best − P̂).

Artifacts: `results/crossbench.md`, `crossbench_dial.py` (prints the dial
table in ~30s from cached data), `crossbench_examples.py` (real side-by-side
example routes).

## Housekeeping

- The OpenRouter key used for labeling lives at `claude/.openrouter_key`
  (gitignored, chmod 600); it passed through chat once — rotate it when this
  experiment cycle ends.
- Total API spend across the whole project: ~$43 ($33 label matrix, ~$10
  LLM-router experiment).
