# Standalone Core Model Router Design

## Objective

Build an isolated experiment under `codex/` that trains a new prompt-to-model
router from public full-information outcomes, evaluates it without proxy or
provider concerns, and serves the resulting policy from a small Rust core.
The experiment must report an unfavorable result as readily as a favorable one.

The primary objective is balanced quality and realized inference cost. The
experiment reports the full quality-cost frontier rather than optimizing one
undocumented scalar preference.

## Scope

Included:

- deterministic public-data ingestion and train/validation/test splitting;
- local prompt embeddings;
- a centroid router trained from prompt-level, per-model outcomes;
- meaningful non-neural and supervised routing baselines;
- validation-only hyperparameter selection;
- frozen test-set quality, cost, regret, and latency metrics;
- a versioned, self-describing artifact;
- Python training/evaluation and Rust inference over precomputed embeddings;
- Python/Rust vector-scoring and decision parity checks;
- a reproducible report containing commands, inputs, hashes, and results.

Excluded:

- HTTP proxying, wire-format translation, provider credentials, billing, and
  session orchestration;
- online learning or production traffic;
- changing `internal/router/cluster` or promoting a router artifact;
- training on RouterArena evaluation data;
- claiming direct superiority over the committed Go router, whose current
  roster lacks a public prompt-level outcome matrix compatible with this
  benchmark.

## Data and provenance

The primary corpus is the LLMRouterBench performance-cost pool. It contains
prompt-level results for 13 flagship models across ten task families, including
quality scores, prompt/completion token counts, and realized cost. The download
is cached under `codex/.cache/` and never committed.

The LLMRouterBench repository identifies itself as MIT, while the Hugging Face
dataset card and archive have no separate license declaration. The experiment
therefore treats the corpus and derived artifact as local research material.
The report records this restriction, and neither raw data nor the trained
artifact is intended for external redistribution without a separate license
review. Source code, synthetic test fixtures, aggregate metrics, and
reproducibility manifests may be committed.

Each raw record is normalized to:

```text
prompt_key, dataset, source_index, prompt, model,
quality, realized_cost, prompt_tokens, completion_tokens
```

The loader constructs one outcome matrix row per unique prompt and one column
per model. Evaluation uses only complete rows shared by the fixed model roster;
the report records rows rejected for missing, duplicate, malformed, or
non-finite outcomes.

Prompt identity is the SHA-256 digest of normalized prompt text. A seeded hash
of prompt identity assigns all copies of a prompt to exactly one split:

- train: 70%;
- validation: 15%;
- test: 15%.

The same digest drives the split across datasets, preventing duplicate prompt
text from crossing split boundaries. The manifest records the data archive
hash, accepted-row digest, dataset and model rosters, split seed, and total
split counts.

## Candidate model

The candidate, `codex-centroid-v1`, uses logical model
`BAAI/bge-small-en-v1.5` from the pinned
`qdrant/bge-small-en-v1.5-onnx-q` snapshot recorded in the manifest. It produces
normalized 384-dimensional vectors locally with no API calls. Python FastEmbed
is used during training; the minimal Rust scorer accepts an already-computed
vector and validates the manifest's embedding identity. Its caller must use
that exact source repository and revision.

For each candidate cluster count, mini-batch k-means fits only training
embeddings. Centroids are L2-normalized after fitting. For every cluster and
model, the trainer estimates expected quality with empirical-Bayes shrinkage:

```text
cluster_quality = (cluster_sum + k0 * global_model_mean) / (cluster_count + k0)
```

At inference, the nearest `top_p` centroids contribute with a softmax over
cosine similarity. Their weighted quality table yields one predicted quality
per eligible model. Model cost is a static expected cost learned exclusively
from training outcomes; realized validation/test completion length is never an
input to routing.

For a requested `quality_bias` in `[0, 1]`, predicted quality and expected cost
are min-max normalized across currently eligible models:

```text
utility = quality_bias * predicted_quality
        + (1 - quality_bias) * (1 - normalized_expected_cost)
```

Ties resolve by manifest model order. The decision includes model, utility,
predicted quality, expected cost, nearest cluster IDs, and artifact identity.

## Hyperparameter selection

The trainer evaluates this fixed search space:

- cluster count: `8, 16, 32, 64`;
- nearest clusters: `1, 2, 4`;
- shrinkage `k0`: `1, 5, 10, 25`;
- softmax temperature: `0.02, 0.05, 0.10`.

Embeddings and k-means fits are cached by immutable input/configuration hashes.
Prepared matrices use length-prefixed UTF-8 prompts and a manifest that binds
the source archive, loader revision, selected paths, roster, audit, and full
matrix digest. Every downstream command revalidates that provenance before use.
Candidate selection uses validation outcomes only. At quality biases
`0.25, 0.50, 0.75`, it calculates realized per-prompt utility from the selected
model's held-out quality and cost and compares it with the per-prompt oracle.
The winning configuration has the lowest mean normalized oracle regret across
the three biases. Stable lexical configuration ordering breaks statistical
ties.

Policy fitting and hyperparameter selection use only train and validation
views. The test view is used for evaluation only after the winning configuration
and artifact are frozen; a regression test verifies that changing all test
outcomes cannot change the selected policy tensors or artifact ID.

## Baselines

All baselines receive the same roster and split:

1. `random`: deterministic seeded selection;
2. `cheapest`: lowest training-set mean realized cost;
3. `best_single`: highest training-set mean quality;
4. `global_utility`: global training quality/cost blend without prompt input;
5. `ridge`: one multi-output ridge regressor over the same BGE embeddings;
6. `knn`: cosine-nearest training prompts with averaged outcome vectors;
7. `oracle`: best held-out model per prompt, used only as an upper bound.

If the public Avengers-Pro adapter can consume the normalized matrix without
changing its algorithm or split, the evaluator runs it in an isolated cache
and records its exact revision. Failure to reproduce it is reported rather
than replaced with a locally altered implementation.

## Metrics

For each method and quality bias, the evaluator reports:

- mean selected quality;
- mean realized cost per request;
- gain over the train-selected best single model;
- normalized gap to the per-prompt oracle;
- cost saving at best-single test quality;
- nondominated quality-cost points and normalized frontier area;
- per-dataset quality, cost, and oracle regret;
- selected-model counts and normalized entropy;
- p50, p95, and p99 pure vector-scoring latency, explicitly excluding embedding;
- paired prompt bootstrap 95% confidence intervals.

The report distinguishes estimated cost used by the policy from realized cost
used for evaluation. A candidate is described as better only when its point is
nondominated by a baseline and the paired quality difference's bootstrap
interval does not cross zero. Otherwise the report says the result is tied,
mixed, or worse.

## Robustness

A deterministic perturbation suite creates capitalization, surrounding
whitespace, punctuation, and semantically neutral wrapper variants for a
sample of held-out prompts. It reports model flip rate plus mean realized
quality and cost changes.
Perturbed prompts never enter training or hyperparameter selection.

## Artifact contract

The trained artifact is a JSON manifest plus compact little-endian float files:

```text
artifact/
├── manifest.json
├── centroids.f32
└── cluster_quality.f32
```

`manifest.json` uses `codex-centroid-artifact-v2` and contains schema version,
artifact ID, training timestamp, stable training provenance, embedding source
and revision, dimensions, roster, model costs, selected hyperparameters,
matrix shapes, file SHA-256 digests, and quality-bias semantics. The artifact
ID binds training and validation inputs but not the held-out test labels or the
timestamp. Search, evaluation, and diagnostic outputs carry the artifact ID,
matrix digest, and seed so the report rejects mixed runs. Loaders reject unknown
schemas or fields, wrong dimensions, non-finite values, hash mismatches,
non-unit centroids, negative costs, and duplicate models.

## Rust core

The Rust workspace exposes a library and CLI. Its public operation is:

```rust
pub fn route_embedding(
    &self,
    embedding: &[f32],
    quality_bias: f64,
    eligible_models: Option<&[String]>,
) -> Result<Decision, RouteError>;
```

The Rust core owns artifact validation, nearest-centroid scoring, cost-quality
blending, deterministic tie-breaking, and decision metadata. Embedding is an
explicit boundary: the caller supplies a vector made with the manifest's exact
model and revision. This keeps the experiment focused on routing policy parity;
end-to-end prompt embedding latency remains future serving integration work.

## Validation and tests

Implementation follows test-first development. Synthetic fixtures establish
the desired behavior before production code is written.

Required gates:

- loader rejects malformed and incomplete outcome matrices;
- duplicate prompts never cross split boundaries;
- preprocessing and splits are deterministic across runs;
- shrinkage converges to the global mean for an empty/sparse cluster;
- selection respects eligibility, bias endpoints, and deterministic ties;
- metric and Pareto calculations match hand-computed fixtures;
- artifact files round-trip and fail on corruption;
- repeated training with the same seed produces identical artifact hashes;
- Python and Rust agree on cluster weights, predicted quality, and model choice
  for at least 20 vectors at every supported bias;
- Rust unit tests, Python tests, formatters, and linters pass;
- a clean-cache end-to-end command downloads, trains, evaluates, and recreates
  the recorded result manifest.

The existing Go routing suite and `cmd/routing-report` run after implementation
to prove that work under `codex/` did not modify established behavior.

## Deliverables

- standalone source under `codex/`;
- locked Python and Rust dependencies;
- documented download, train, evaluate, and route commands;
- one trained local `codex-centroid-v1` artifact;
- machine-readable test metrics and hyperparameter search results;
- a concise Markdown comparison report;
- provenance and reproducibility manifests;
- evidence from fresh Python, Rust, parity, and end-to-end verification runs.
