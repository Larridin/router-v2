# Codex centroid router lab

This directory is a standalone, reproducible experiment for the core question:
given a prompt and a set of eligible LLMs, can a small semantic router choose a
better quality/cost point than static model selection?

It deliberately contains no HTTP proxy, authentication, retries, billing, or
provider dispatch. Python prepares data, trains, searches, and evaluates. Rust
independently validates the frozen artifact and implements the serving-time
vector scorer.

## Result

The frozen `codex-centroid-v1` candidate was selected on validation only and
evaluated on 1,811 held-out prompts only after selection. Its artifact ID is
`c2c9c71eac592a304b9d1e2bcc7818ed59bf739ae6e45fc191dfc2888e8f7b8e`.
The Weave comparator retrains the v0.75 core cluster algorithm on exactly the
same training rows, embeddings, 13-model roster, and costs.

| Quality bias | Codex quality | Weave quality | Best single | Codex gain | Weave gain | Codex cost | Weave cost | Paired Codex-Weave quality CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.5897 | 0.5756 | 0.5986 | -0.0088 | -0.0229 | $0.01191 | $0.01112 | [+0.0039, +0.0248] |
| 0.75 | 0.6057 | 0.6082 | 0.5986 | +0.0072 | +0.0097 | $0.02312 | $0.02337 | [-0.0075, +0.0022] |
| 1.00 | 0.6162 | 0.6209 | 0.5986 | +0.0177 | +0.0224 | $0.03692 | $0.02893 | [-0.0163, +0.0069] |

Codex does not beat Weave overall. At bias 1.0, both beat best single with
95% intervals above zero, but Weave has the larger numerical gain and costs
21.6% less than Codex. Their direct quality difference is inconclusive. Codex
does beat Weave by 0.0141 quality at bias 0.5 with a paired interval above
zero and similar cost, but both routers are below best single there. The
frontiers cross; Weave owns the stronger high-quality endpoint.

Surface-form robustness changed 2 of 1,024 decisions (0.195%). The pure Python
vector scorer measured 18.1 microseconds p50 and 20.7 microseconds p95 on this
machine, excluding embedding. See [results/REPORT.md](results/REPORT.md) for the
full bias curve, baselines, per-dataset results, confidence intervals, and
caveats.

## How the model works

1. Normalize prompt line endings and hash prompt text for a stable 70/15/15
   train/validation/test split. Duplicate prompt text cannot cross splits.
2. Embed prompts locally with logical model `BAAI/bge-small-en-v1.5`, using the
   ONNX files from `qdrant/bge-small-en-v1.5-onnx-q` at revision
   `52398278842ec682c6f32300af41344b1c0b0bb2`, producing normalized 384-d vectors.
3. Fit MiniBatch K-means on training vectors only.
4. For every cluster and model, estimate mean quality with empirical-Bayes
   shrinkage toward that model's global training mean.
5. At inference, blend the nearest `top_p` cluster quality rows with a softmax
   over cosine similarity. Normalize predicted quality and train-derived cost
   across eligible models, blend them using the caller's `quality_bias`, and
   choose the first maximum in manifest order.

The fixed validation search covered 144 configurations:

- clusters: 8, 16, 32, 64;
- nearest centroids: 1, 2, 4;
- shrinkage: 1, 5, 10, 25;
- temperature: 0.02, 0.05, 0.10.

It minimized mean normalized oracle regret at quality biases 0.25, 0.50, and
0.75. The winner was 8 clusters, top-1, shrinkage 1, temperature 0.02. Test
outcomes play no part in that choice; a regression test changes every hidden
test label and proves the selected policy tensors and artifact ID remain
identical.

The Weave comparator uses full K-means with 16 clusters and 10 initializations,
L2-normalizes the centroids, reassigns training rows by cosine, and equally
weights the four nearest centroids. It z-scores quality across models per
training prompt, clips to `[-3, 3]`, maps to `[0, 1]`, applies empirical-Bayes
shrinkage of 10, and min-max normalizes cluster quality and global mean training
cost. The shared sweep uses quality bias directly as uniform alpha in the same
quality-cost blend. Production roster-specific dial calibration, alpha floors,
provider/capability filtering, and subsidies are outside this core comparison.

The Weave hyperparameters are fixed rather than tuned on this public data;
Codex received a 144-configuration validation search, so the tuning budget
favors Codex. The trainer behavior is pinned to repository revision
`b74af8941cb71e3a0bc43af9d9a68b0c727fb7cf`. K-means runs with one numerical
thread for byte-reproducible artifacts. The frozen Weave policy SHA-256 is
`da38fdb22e44d5b4bc74cb2d0969915a55429c043b1da897fb33bbe4d691c141`;
two complete evaluations produced identical result bytes.

## Public data and restrictions

The input is [NPULH/LLMRouterBench](https://huggingface.co/datasets/NPULH/LLMRouterBench),
using its performance/cost flagship pool. The pinned release is:

- URL: `https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/main/bench-release.tar.gz`
- size: 1,283,503,080 bytes;
- SHA-256: `b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1`.

Preparation accepted 12,166 complete prompts and 161,520 raw outcomes for 13
models. It rejected 282 prompts with incomplete rosters; this removes all of
tau2 because the release has only 12 of the 13 requested flagship models for
that dataset. Ten failed-call negative completion-token sentinels were changed
to zero and explicitly audited; no score or cost was imputed.

The upstream GitHub repository identifies itself as MIT, but the Hugging Face
dataset card/archive has no separate license declaration. Raw JSON, prompts,
embeddings, and the derived artifact therefore remain local and gitignored.
Only aggregate metrics and provenance are committed here. Confirm your own
redistribution rights before publishing derived prompt-level data or weights.

Expect roughly 10 GB of free disk and budget 12 GB of RAM. On the development
machine, clean preparation peaked at 3.09 GiB resident memory and clean local
embedding/training peaked at 9.07 GiB. The prepared matrix is 13 MB compressed
and about 49 MB expanded because prompt strings use length-prefixed UTF-8 rather
than fixed-width Unicode; a regression test prevents padded-string blowups.

## Reproduce

Prerequisites are Python 3.12, [uv](https://docs.astral.sh/uv/), Rust 1.90 or
newer, and about 10 GB free disk. All embedding inference is local.

Bootstrap and unit tests:

```bash verify
cd codex
uv sync --frozen
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
```

Run each immutable stage. `evaluate` refuses to replace an existing frozen
test result, and `diagnose` likewise refuses to replace diagnostics.

```bash verify
cd codex
uv run routerlab download --cache-root .cache --artifact-root artifacts --results-root results --seed 42
uv run routerlab prepare --cache-root .cache --artifact-root artifacts --results-root results --seed 42
uv run routerlab train --cache-root .cache --artifact-root artifacts --results-root results --seed 42
uv run routerlab evaluate --cache-root .cache --artifact-root artifacts --results-root results --seed 42
uv run routerlab diagnose --cache-root .cache --artifact-root artifacts --results-root results --seed 42
uv run routerlab report --cache-root .cache --artifact-root artifacts --results-root results --seed 42
```

For a fresh repeat while retaining the first run, use a new run ID:

```bash verify
cd codex
uv run routerlab run-all --cache-root .cache --artifact-root artifacts --results-root results --seed 42 --new-run-id repeat-01
```

The first uncached preparation takes several minutes because the release is
hundreds of large JSON files. Initial local embedding took about ten minutes on
the development machine. Both normalized outcomes and embeddings are cached.

## Artifact and Rust serving core

The local v2 artifact directory contains canonical `manifest.json`, little-endian
`centroids.f32`, and `cluster_quality.f32`. The manifest binds model order,
logical embedding model, ONNX source repo and revision, stable training context,
configuration, expected costs, shapes, and tensor SHA-256 digests into one
content-addressed artifact ID. Training time is provenance and does not alter
that ID.

The Rust binary accepts one already-computed embedding as JSON on stdin. It
revalidates schema, content identity, tensor hashes, dimensions, finite values,
and model uniqueness before routing. Prompt embedding is intentionally outside
this minimal serving core; the caller must use the manifest's exact BGE model.

```json
{"embedding":[0.1,0.2],"quality_bias":0.75,"eligible_models":["model-a","model-b"]}
```

Rust and cross-language verification:

```bash verify
cd codex/rust
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cd ..
uv run pytest tests/test_rust_parity.py -q
```

Python/Rust parity covers 20 randomized embeddings, all five bias settings,
cluster order, softmax weights, model index, and final decision. The Rust core
also successfully loads and routes the real artifact.

## What is and is not comparable

Baselines use only training data: random, cheapest, best single, global utility,
multi-output ridge, exact cosine kNN, and the Weave v0.75 cluster algorithm.
Oracle sees held-out outcomes and is an evaluator-only upper bound.

The Weave comparison is exact for the scoped core algorithm: its recipe is
retrained on the same public matrix and evaluated on the same hidden prompts
without proxy model mappings. It is not a literal replay of the deployed v0.75
artifact, which targets a newer 18-model roster and Jina-768 embeddings and
adds the production-only policies excluded above. That distinction does not
prevent the intended same-data comparison of routing algorithms, but it does
prevent claiming these numbers reproduce production traffic.

The benchmark is offline counterfactual evaluation over historical model calls.
It does not measure current provider behavior, live failure rates, latency,
prompt-cache effects, or production traffic distribution. Before deployment,
retrain on a current target roster and run an online shadow/bake-off evaluation.
