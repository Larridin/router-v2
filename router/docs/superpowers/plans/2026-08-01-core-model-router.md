# Standalone Core Model Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, train, and honestly evaluate a standalone centroid model router under `codex/`, with Python training and a parity-tested Rust inference core.

**Architecture:** Python ingests a full-information prompt/model outcome matrix, creates leakage-safe hash splits, embeds prompts with pinned BGE-small ONNX, trains and selects centroid configurations on validation regret, and evaluates a frozen candidate against common baselines. It exports hashed binary tensors plus a JSON manifest that an independent Rust library validates and serves.

**Tech Stack:** Python 3.12, uv, NumPy, SciPy, scikit-learn, FastEmbed, pytest, Ruff; Rust 1.90, Cargo, serde, clap, sha2, thiserror.

---

## File map

- `codex/pyproject.toml`: locked Python package, CLI, and tooling configuration.
- `codex/.python-version`: pinned interpreter line.
- `codex/.gitignore`: excludes downloaded data, embedding cache, and local artifacts.
- `codex/README.md`: exact download, training, evaluation, and Rust routing commands.
- `codex/src/routerlab/schema.py`: immutable normalized matrix and split value types.
- `codex/src/routerlab/data.py`: archive download, digest verification, JSON discovery, and matrix construction.
- `codex/src/routerlab/split.py`: deterministic prompt-group hash split.
- `codex/src/routerlab/embeddings.py`: pinned FastEmbed adapter and content-addressed cache.
- `codex/src/routerlab/centroid.py`: candidate fit, prediction, and hyperparameter search.
- `codex/src/routerlab/baselines.py`: random, cheapest, best-single, global, ridge, kNN, and oracle policies.
- `codex/src/routerlab/metrics.py`: quality/cost/regret/frontier/bootstrap calculations.
- `codex/src/routerlab/artifact.py`: binary tensor and manifest serialization/validation.
- `codex/src/routerlab/pipeline.py`: train/validation freeze/test orchestration.
- `codex/src/routerlab/cli.py`: user-facing commands.
- `codex/src/routerlab/report.py`: deterministic Markdown report renderer.
- `codex/tests/fixtures/`: tiny hand-verifiable outcome matrices and prompts.
- `codex/tests/test_*.py`: Python behavior tests.
- `codex/rust/Cargo.toml`: standalone Rust crate and CLI.
- `codex/rust/src/artifact.rs`: manifest and tensor validation.
- `codex/rust/src/router.rs`: pure scoring and selection.
- `codex/rust/src/embedder.rs`: BGE-small FastEmbed wrapper.
- `codex/rust/src/lib.rs`: public core API.
- `codex/rust/src/main.rs`: JSON CLI.
- `codex/rust/tests/`: artifact, route, and Python parity tests.
- `codex/results/`: aggregate search, test, robustness, latency, and provenance outputs.

### Task 1: Bootstrap an isolated tested Python package

**Files:**
- Create: `codex/pyproject.toml`
- Create: `codex/.python-version`
- Create: `codex/.gitignore`
- Create: `codex/src/routerlab/__init__.py`
- Create: `codex/tests/test_package.py`

- [ ] **Step 1: Write the failing package smoke test**

```python
def test_package_exposes_version() -> None:
    import routerlab

    assert routerlab.__version__ == "0.1.0"
```

- [ ] **Step 2: Run it and verify import failure**

Run: `cd codex && uv run pytest tests/test_package.py -q`

Expected: failure because `routerlab` does not exist.

- [ ] **Step 3: Add package metadata and the minimal version module**

Use Python `>=3.12,<3.13`; runtime dependencies are `numpy`, `scipy`,
`scikit-learn`, `fastembed`, and `httpx`. Development dependencies are
`pytest`, `pytest-cov`, and `ruff`. Configure the `routerlab` console script to
call `routerlab.cli:main` and configure Ruff for Python 3.12.

```python
"""Standalone model-routing research package."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Lock dependencies and verify the test passes**

Run: `cd codex && uv lock && uv run pytest tests/test_package.py -q`

Expected: one passing test.

- [ ] **Step 5: Commit the package scaffold**

```bash
git add codex/pyproject.toml codex/uv.lock codex/.python-version codex/.gitignore codex/src/routerlab/__init__.py codex/tests/test_package.py
git commit -m "feat(codex): scaffold standalone router lab"
```

### Task 2: Normalize outcomes and create leakage-safe splits

**Files:**
- Create: `codex/src/routerlab/schema.py`
- Create: `codex/src/routerlab/data.py`
- Create: `codex/src/routerlab/split.py`
- Create: `codex/tests/fixtures/tiny_outcomes.json`
- Create: `codex/tests/test_data.py`
- Create: `codex/tests/test_split.py`

- [ ] **Step 1: Write failing tests for matrix construction**

The fixture contains three prompts, three models, and one intentionally
incomplete prompt. Assert that matrix construction returns the two complete
rows, stable lexical model order, float arrays of shape `(2, 3)`, and one
explicit rejection reason.

```python
matrix, audit = load_outcome_files([fixture_path])
assert matrix.models == ("model-a", "model-b", "model-c")
assert matrix.quality.shape == (2, 3)
assert matrix.realized_cost.shape == (2, 3)
assert audit.rejections == {"incomplete_roster": 1}
```

- [ ] **Step 2: Run the data tests and observe missing-module failure**

Run: `cd codex && uv run pytest tests/test_data.py -q`

- [ ] **Step 3: Implement immutable matrix types and loader validation**

`OutcomeMatrix` carries prompt keys, dataset names, source indices, prompts,
models, quality, cost, prompt tokens, and completion tokens. Validate finite
numeric values, unique `(prompt_key, model)` pairs, equal tensor shapes, and a
complete shared roster. Normalize line endings and surrounding whitespace
before hashing prompt text.

- [ ] **Step 4: Run data tests to green**

Run: `cd codex && uv run pytest tests/test_data.py -q`

- [ ] **Step 5: Write failing split tests**

Assert deterministic 70/15/15 membership, seed sensitivity, and that repeated
prompt text appearing under different datasets always receives one split.

```python
first = assign_splits(matrix, seed=42)
second = assign_splits(matrix, seed=42)
assert first.tolist() == second.tolist()
assert one_split_per_prompt(matrix.prompt_keys, first)
```

- [ ] **Step 6: Implement SHA-256 bucket splitting and run tests**

Hash `b"codex-router-split-v1\0" + seed_bytes + prompt_digest`, interpret the
first eight bytes as an unsigned big-endian integer, and map modulo 10,000 to
train `<7000`, validation `<8500`, and test otherwise.

Run: `cd codex && uv run pytest tests/test_split.py -q`

- [ ] **Step 7: Commit normalized data behavior**

```bash
git add codex/src/routerlab/schema.py codex/src/routerlab/data.py codex/src/routerlab/split.py codex/tests/fixtures/tiny_outcomes.json codex/tests/test_data.py codex/tests/test_split.py
git commit -m "feat(codex): normalize outcomes with leakage-safe splits"
```

### Task 3: Add deterministic local embeddings and caching

**Files:**
- Create: `codex/src/routerlab/embeddings.py`
- Create: `codex/tests/test_embeddings.py`

- [ ] **Step 1: Write failing cache tests with a real deterministic fake encoder**

Use a small encoder object whose output is derived from SHA-256 bytes. Assert
that duplicate prompts are encoded once, rows are normalized, a second call
loads the `.npy` cache without invoking the encoder, and changing model revision
changes the cache key.

- [ ] **Step 2: Run and observe failure**

Run: `cd codex && uv run pytest tests/test_embeddings.py -q`

- [ ] **Step 3: Implement `EmbeddingCache` and `FastEmbedEncoder`**

Pin logical model ID `BAAI/bge-small-en-v1.5`, source repository
`qdrant/bge-small-en-v1.5-onnx-q`, dimension `384`, and an explicit source
revision in the cache manifest. Store `float32` arrays and reject wrong shapes,
non-finite values, zero norms, prompt-hash mismatches, and manifest drift.

- [ ] **Step 4: Run embedding tests to green**

Run: `cd codex && uv run pytest tests/test_embeddings.py -q`

- [ ] **Step 5: Commit embedding support**

```bash
git add codex/src/routerlab/embeddings.py codex/tests/test_embeddings.py
git commit -m "feat(codex): add content-addressed prompt embeddings"
```

### Task 4: Implement the centroid policy test-first

**Files:**
- Create: `codex/src/routerlab/centroid.py`
- Create: `codex/tests/test_centroid.py`

- [ ] **Step 1: Write failing shrinkage and scoring tests**

Use two orthogonal centroids and three models with hand-computed outcomes.
Assert exact shrinkage, softmax weights, bias endpoints, eligibility filtering,
and manifest-order tie breaking.

```python
decision = policy.route_vector(
    np.array([1.0, 0.0], dtype=np.float32),
    eligible_models=("model-a", "model-b"),
    quality_bias=1.0,
)
assert decision.model == "model-b"
assert decision.cluster_ids == (0,)
```

- [ ] **Step 2: Run and observe failure**

Run: `cd codex && uv run pytest tests/test_centroid.py -q`

- [ ] **Step 3: Implement fitting and pure vector routing**

Fit `MiniBatchKMeans` only on training embeddings, normalize centroids, derive
global model means/costs, assign training rows to nearest centroids, and apply
the design's shrinkage equation. Validate `top_p`, temperature, dimensions,
eligibility, and bias range.

- [ ] **Step 4: Run centroid tests to green**

Run: `cd codex && uv run pytest tests/test_centroid.py -q`

- [ ] **Step 5: Add deterministic-fit regression test**

Fit the same synthetic matrix twice with seed 42 and assert every tensor is
bitwise identical.

- [ ] **Step 6: Commit the centroid policy**

```bash
git add codex/src/routerlab/centroid.py codex/tests/test_centroid.py
git commit -m "feat(codex): train and score centroid routing policy"
```

### Task 5: Build honest routing baselines

**Files:**
- Create: `codex/src/routerlab/baselines.py`
- Create: `codex/tests/test_baselines.py`

- [ ] **Step 1: Write failing behavior tests for every baseline**

On a hand-computed matrix, assert random reproducibility, cheapest and
best-single train-only selection, global utility endpoints, multi-output ridge
prediction shape, kNN exclusion of validation/test rows, and oracle selection
from held-out outcomes only.

- [ ] **Step 2: Run and observe failure**

Run: `cd codex && uv run pytest tests/test_baselines.py -q`

- [ ] **Step 3: Implement baseline policies behind one protocol**

```python
class VectorPolicy(Protocol):
    name: str

    def select(self, embeddings: NDArray[np.float32], quality_bias: float) -> NDArray[np.int64]:
        raise NotImplementedError
```

Fit global statistics, `Ridge(alpha=1.0)`, and cosine `NearestNeighbors` from
training rows only. The oracle is evaluator-only and cannot serialize as a
serving artifact.

- [ ] **Step 4: Run baseline tests to green and commit**

Run: `cd codex && uv run pytest tests/test_baselines.py -q`

```bash
git add codex/src/routerlab/baselines.py codex/tests/test_baselines.py
git commit -m "feat(codex): add comparable routing baselines"
```

### Task 6: Implement metrics, Pareto analysis, and bootstrap intervals

**Files:**
- Create: `codex/src/routerlab/metrics.py`
- Create: `codex/tests/test_metrics.py`

- [ ] **Step 1: Write failing hand-calculated metric tests**

Assert selected quality/cost, best-single gain, oracle gap, model entropy,
nondominated frontier membership, normalized frontier area, and paired
bootstrap determinism on tiny arrays.

- [ ] **Step 2: Run and observe failure**

Run: `cd codex && uv run pytest tests/test_metrics.py -q`

- [ ] **Step 3: Implement metric functions without model-specific behavior**

Reject invalid selection indices and non-finite matrices. Pareto dominance
requires quality greater-or-equal and cost less-or-equal with at least one
strict inequality. Bootstrap resamples prompt rows with seed 42 and 2,000
replicates.

- [ ] **Step 4: Run metric tests to green and commit**

Run: `cd codex && uv run pytest tests/test_metrics.py -q`

```bash
git add codex/src/routerlab/metrics.py codex/tests/test_metrics.py
git commit -m "feat(codex): measure router quality cost and regret"
```

### Task 7: Export and validate immutable artifacts

**Files:**
- Create: `codex/src/routerlab/artifact.py`
- Create: `codex/tests/test_artifact.py`

- [ ] **Step 1: Write failing round-trip and corruption tests**

Export a synthetic policy, reload it, and assert identical decisions. Then
change one byte in each tensor and assert a SHA-256 mismatch. Add failures for
unknown schema, duplicate model, wrong shape, and NaN.

- [ ] **Step 2: Run and observe failure**

Run: `cd codex && uv run pytest tests/test_artifact.py -q`

- [ ] **Step 3: Implement schema `codex-centroid-artifact-v2`**

Write little-endian contiguous `float32` tensors and canonical JSON with sorted
keys and no timestamp in the content-addressed artifact ID. Bind stable training
and validation provenance, embedding identity, tensor hashes, costs, and policy
configuration. Store the training timestamp as provenance but exclude it and
all held-out test labels from artifact identity.

- [ ] **Step 4: Run artifact tests to green and commit**

Run: `cd codex && uv run pytest tests/test_artifact.py -q`

```bash
git add codex/src/routerlab/artifact.py codex/tests/test_artifact.py
git commit -m "feat(codex): export validated centroid artifacts"
```

### Task 8: Orchestrate search, test evaluation, robustness, and reports

**Files:**
- Create: `codex/src/routerlab/pipeline.py`
- Create: `codex/src/routerlab/report.py`
- Create: `codex/src/routerlab/cli.py`
- Create: `codex/tests/test_pipeline.py`
- Create: `codex/tests/test_report.py`

- [ ] **Step 1: Write a failing synthetic end-to-end test**

Run the complete pipeline on a generated separable corpus. Assert that only
validation metrics select hyperparameters, the frozen artifact predates the
test result file in pipeline state, the centroid beats random on test quality,
and rerunning produces identical artifact and result hashes.

- [ ] **Step 2: Run and observe failure**

Run: `cd codex && uv run pytest tests/test_pipeline.py -q`

- [ ] **Step 3: Implement the fixed search and freeze boundary**

Fit embeddings once; fit k-means once per cluster count; evaluate all declared
`top_p`, shrinkage, and temperature combinations at biases 0.25, 0.50, and
0.75; select minimum validation oracle regret; serialize the artifact; then
evaluate test outcomes and baselines. Record every configuration, not only the
winner.

- [ ] **Step 4: Implement deterministic robustness perturbations**

Use whitespace, capitalization, terminal punctuation, and `Please answer:`
wrappers over a seed-selected held-out subset. Report flip rate and utility
delta without adding perturbed rows to fit/search.

- [ ] **Step 5: Write and pass report snapshot tests**

The Markdown report includes provenance, split counts, winner, baseline table,
frontier table, per-dataset table, robustness, latency, caveats, and exact
reproduction commands.

Run: `cd codex && uv run pytest tests/test_pipeline.py tests/test_report.py -q`

- [ ] **Step 6: Add CLI commands and commit**

Commands: `download`, `prepare`, `train`, `evaluate`, `report`, and `run-all`.
Each accepts an explicit cache/results root and refuses to overwrite a frozen
test result unless `--new-run-id` is supplied.

```bash
git add codex/src/routerlab/pipeline.py codex/src/routerlab/report.py codex/src/routerlab/cli.py codex/tests/test_pipeline.py codex/tests/test_report.py
git commit -m "feat(codex): orchestrate reproducible router experiments"
```

### Task 9: Implement the Rust artifact loader and pure scorer

**Files:**
- Create: `codex/rust/Cargo.toml`
- Create: `codex/rust/src/artifact.rs`
- Create: `codex/rust/src/router.rs`
- Create: `codex/rust/src/lib.rs`
- Create: `codex/rust/tests/artifact.rs`
- Create: `codex/rust/tests/router.rs`

- [ ] **Step 1: Write failing Rust artifact tests**

Load a Python-generated synthetic fixture and test schema, dimensions, SHA-256,
finite floats, and duplicate model failures.

Run: `cd codex/rust && cargo test --test artifact`

Expected: compilation failure because the library does not exist.

- [ ] **Step 2: Implement the minimal validated loader**

Use `serde`, `serde_json`, `sha2`, `bytemuck`, and `thiserror`. Decode
little-endian float tensors explicitly rather than reinterpreting native bytes.

- [ ] **Step 3: Write failing pure scorer tests**

Use a hand-written vector and synthetic artifact to assert the same endpoint,
eligibility, top-p, softmax, and tie behavior as Python.

- [ ] **Step 4: Implement `Router::route_embedding` and pass tests**

Run: `cd codex/rust && cargo test --test artifact --test router`

- [ ] **Step 5: Commit the pure Rust core**

```bash
git add codex/rust/Cargo.toml codex/rust/Cargo.lock codex/rust/src codex/rust/tests
git commit -m "feat(codex): load and score router artifacts in rust"
```

### Task 10: Add the Rust vector CLI and cross-language scorer parity

**Files:**
- Create: `codex/rust/src/main.rs`
- Create: `codex/tests/test_rust_parity.py`

- [ ] **Step 1: Write failing Rust vector CLI tests**

The CLI reads one JSON request from stdin and emits one JSON decision. Assert
invalid bias/eligibility errors are structured and never produce a model.

- [ ] **Step 2: Implement the validated vector-scoring CLI**

Accept an already-computed vector, normalize defensively, verify dimensions and
finite values, and return artifact identity with the decision. Add no HTTP server.

- [ ] **Step 3: Write failing Python/Rust parity tests**

For at least 20 deterministic vectors, assert identical cluster weights,
predicted quality, and model decisions at biases `0.0, 0.25, 0.5, 0.75, 1.0`.

- [ ] **Step 4: Resolve only implementation differences and pass parity**

Run: `cd codex && uv run pytest tests/test_rust_parity.py -q`

- [ ] **Step 5: Commit inference parity**

```bash
git add codex/rust/src/main.rs codex/tests/test_rust_parity.py
git commit -m "feat(codex): add parity-tested rust vector routing"
```

### Task 11: Download public outcomes and train the real candidate

**Files:**
- Create locally: `codex/.cache/llmrouterbench/bench-release.tar.gz`
- Create locally: `codex/.cache/llmrouterbench/results/`
- Create locally: `codex/artifacts/codex-centroid-v1/`
- Create: `codex/results/search.json`
- Create: `codex/results/test_metrics.json`
- Create: `codex/results/robustness.json`
- Create: `codex/results/provenance.json`
- Create: `codex/results/REPORT.md`

- [ ] **Step 1: Download and record the archive digest**

Run: `cd codex && uv run routerlab download --cache-root .cache`

Expected: HTTPS download, SHA-256 printed and stored, archive safely extracted
without absolute paths or `..` entries.

- [ ] **Step 2: Prepare and audit the full outcome matrix**

Run: `cd codex && uv run routerlab prepare --cache-root .cache --results-root results`

Inspect accepted/rejected counts, roster, per-dataset completeness, and split
overlap assertion before training.

- [ ] **Step 3: Train/search on train and validation only**

Run: `cd codex && uv run routerlab train --cache-root .cache --artifact-root artifacts --results-root results --seed 42`

Expected: every declared configuration in `results/search.json`, one selected
configuration, and a content-addressed local artifact.

- [ ] **Step 4: Evaluate only after freezing the selected artifact**

Run: `cd codex && uv run routerlab evaluate --cache-root .cache --artifact-root artifacts --results-root results --seed 42`

Expected: baseline, frontier, dataset, and bootstrap files linked to the frozen
artifact ID. Reproducible reruns are allowed; test outcomes never feed fitting
or model selection.

- [ ] **Step 5: Render and inspect the report**

Run: `cd codex && uv run routerlab report --results-root results`

Check that claims follow confidence intervals and that the license, roster, and
cross-router-comparison limitations are explicit.

- [ ] **Step 6: Commit aggregate reproducibility results only**

Do not commit raw outcomes, prompt text, embeddings, or the locally restricted
artifact.

```bash
git add codex/results/search.json codex/results/test_metrics.json codex/results/robustness.json codex/results/provenance.json codex/results/REPORT.md
git commit -m "eval(codex): report standalone router benchmark"
```

### Task 12: Documentation and final verification

**Files:**
- Create: `codex/README.md`
- Modify: `codex/pyproject.toml`

- [ ] **Step 1: Write README command-validation test**

Extract fenced shell commands tagged `verify` and assert every local command is
recognized by the Python or Rust CLI. This prevents documentation drift.

- [ ] **Step 2: Document prerequisites, data restrictions, and exact workflow**

Include disk requirements, cache locations, archive digest, clean-cache run,
individual stages, artifact format, Rust CLI request/response, test commands,
and interpretation of every metric.

- [ ] **Step 3: Run Python verification**

Run: `cd codex && uv run ruff format --check . && uv run ruff check . && uv run pytest -q`

Expected: all checks pass with no warnings.

- [ ] **Step 4: Run Rust verification**

Run: `cd codex/rust && cargo fmt --check && cargo clippy --all-targets --all-features -- -D warnings && cargo test --all-features`

Expected: formatting, lint, and tests pass.

- [ ] **Step 5: Run reproducibility verification**

Run the synthetic clean-cache pipeline twice into two temporary directories and
compare artifact and metrics hashes. Then route the parity corpus through both
Python and Rust.

- [ ] **Step 6: Prove the existing router remains green**

Run: `go test -count=1 ./internal/router/...`

Run: `go run -tags no_onnx ./cmd/routing-report --target v0.75 --baseline v0.73`

- [ ] **Step 7: Commit documentation and inspect final diff**

```bash
git add codex/README.md codex/pyproject.toml codex/uv.lock
git commit -m "docs(codex): document reproducible router training"
git status --short
git log --oneline --max-count=15
```
