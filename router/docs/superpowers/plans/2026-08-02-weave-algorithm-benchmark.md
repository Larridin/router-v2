# Weave Algorithm Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a leakage-safe, algorithm-level `weave_v075` comparator to the Codex public-data router experiment and report its relative gain against the new router.

**Architecture:** A new pure Python module fits the fixed Weave v0.75 cluster recipe from the existing training matrix and BGE vectors. The current evaluator treats it as another vector policy, but additionally emits direct paired Codex-versus-Weave evidence and provenance that the report renders explicitly.

**Tech Stack:** Python 3.12, NumPy, scikit-learn, pytest, Ruff; existing routerlab metrics and result schemas.

---

### Task 1: Specify and test the pure Weave policy

**Files:**
- Create: `codex/src/routerlab/weave.py`
- Create: `codex/tests/test_weave.py`

- [ ] **Step 1: Write failing tests for the exact math**

Add hand-calculated tests that require `zscore_per_prompt`, fixed
`WeaveConfig.v075()`, shrunk cluster means, equal top-P summation, global
train-cost normalization, stable ties, and movement from cheapest at alpha 0
to predicted-best at alpha 1.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `cd codex && uv run pytest -q tests/test_weave.py`

Expected: collection fails because `routerlab.weave` does not exist.

- [ ] **Step 3: Implement the minimal pure policy**

Implement `WeaveConfig`, `WeavePolicy`, `zscore_per_prompt`,
`fit_weave_policy`, and a deterministic policy digest. Use normalized vectors,
full `sklearn.cluster.KMeans`, train-only quality/cost, first-index argmax ties,
and no I/O.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `cd codex && uv run pytest -q tests/test_weave.py`

Expected: every Weave unit test passes.

- [ ] **Step 5: Commit the pure policy**

```bash
git add codex/src/routerlab/weave.py codex/tests/test_weave.py
git commit -m "feat(codex): add faithful weave cluster baseline"
```

### Task 2: Integrate leakage-safe paired evaluation

**Files:**
- Modify: `codex/src/routerlab/pipeline.py`
- Modify: `codex/tests/test_pipeline.py`

- [ ] **Step 1: Write failing pipeline assertions**

Extend the synthetic end-to-end test to require `weave_v075` points, a
`weave_comparison` entry for every bias, fixed configuration provenance, a
policy digest, and paired quality/cost intervals. Add a regression that mutates
all test outcomes and asserts the Weave digest and selections remain unchanged.

- [ ] **Step 2: Run the focused pipeline tests and verify RED**

Run: `cd codex && uv run pytest -q tests/test_pipeline.py`

Expected: assertions fail because the evaluator has no Weave results.

- [ ] **Step 3: Fit and evaluate Weave from training rows**

Fit `WeavePolicy` after the split from `train_vectors`, `train.quality`, and
`train.realized_cost`. Add it to the common policy loop. Emit per-router
best-single gains and paired Codex-minus-Weave bootstrap intervals for each
quality bias, plus separate Codex and Weave cost saving at reference quality.

- [ ] **Step 4: Run pipeline tests and verify GREEN**

Run: `cd codex && uv run pytest -q tests/test_pipeline.py`

Expected: all pipeline tests pass and no test outcomes affect fitted policy
identity.

- [ ] **Step 5: Commit evaluation integration**

```bash
git add codex/src/routerlab/pipeline.py codex/tests/test_pipeline.py
git commit -m "feat(codex): compare routers on identical held-out data"
```

### Task 3: Render the algorithm comparison

**Files:**
- Modify: `codex/src/routerlab/report.py`
- Modify: `codex/tests/test_report.py`
- Modify: `codex/README.md`

- [ ] **Step 1: Write a failing report test**

Require a `Codex versus Weave algorithm` section with shared-control wording,
both gains over best single, direct quality/cost deltas, and confidence bounds.

- [ ] **Step 2: Run the report tests and verify RED**

Run: `cd codex && uv run pytest -q tests/test_report.py`

Expected: the comparison heading and rows are absent.

- [ ] **Step 3: Render direct evidence and document interpretation**

Add the comparison section before generic baselines. Keep claims mechanical:
print estimates and intervals, and describe significance only when interval
bounds justify it. Update README methodology after real results exist.

- [ ] **Step 4: Run report tests and verify GREEN**

Run: `cd codex && uv run pytest -q tests/test_report.py`

Expected: all report tests pass.

- [ ] **Step 5: Commit reporting changes**

```bash
git add codex/src/routerlab/report.py codex/tests/test_report.py codex/README.md
git commit -m "docs(codex): report weave algorithm comparison"
```

### Task 4: Regenerate and verify the public benchmark

**Files:**
- Modify: `codex/results/test_metrics.json`
- Modify: `codex/results/REPORT.md`
- Modify: `codex/README.md`

- [ ] **Step 1: Preserve the previous frozen outputs locally**

Move the current test metrics, diagnostics, and report to a new explicit folder
under `codex/.cache/`; do not delete them.

- [ ] **Step 2: Re-run evaluation, diagnostics, and report**

Run:

```bash
cd codex
uv run routerlab evaluate --cache-root .cache --artifact-root artifacts --results-root results --seed 42
uv run routerlab diagnose --cache-root .cache --artifact-root artifacts --results-root results --seed 42
uv run routerlab report --results-root results
```

Expected: the Codex artifact ID remains unchanged, while test results include
`weave_v075` and direct paired comparisons.

- [ ] **Step 3: Update README with measured results**

State whether Codex beats, ties, or loses to Weave at each useful frontier
point. Include relative gains over best single and paired confidence intervals;
do not generalize beyond the public benchmark.

- [ ] **Step 4: Run all verification gates**

Run:

```bash
cd codex
uv run ruff format --check .
uv run ruff check .
uv run pytest -q
cargo fmt --manifest-path rust/Cargo.toml --check
cargo clippy --manifest-path rust/Cargo.toml --all-targets --all-features -- -D warnings
cargo test --manifest-path rust/Cargo.toml --all-targets --all-features
cd ..
go test -count=1 ./internal/router/...
go run -tags no_onnx ./cmd/routing-report --target v0.75 --baseline v0.73
```

Expected: every command exits zero.

- [ ] **Step 5: Commit aggregate evidence**

```bash
git add codex/README.md codex/results/test_metrics.json codex/results/REPORT.md
git commit -m "eval(codex): benchmark against weave cluster algorithm"
```
