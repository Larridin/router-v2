# Weave Algorithm Benchmark Design

## Objective

Benchmark the new `codex_centroid` policy against the algorithm behind Weave's
v0.75 cluster router, with both policies retrained and evaluated on the same
public LLMRouterBench matrix. This is an algorithm comparison, not a comparison
of deployed artifacts or model names.

## Selected approach

Implement a first-class `weave_v075` evaluation policy inside `codex/`. It uses
the exact core recipe declared by
`internal/router/cluster/artifacts/v0.75/metadata.yaml` and the v2 scorer:

- full KMeans with `K=16`, `n_init=10`, and seed 42, followed by centroid
  L2-normalization and cosine reassignment of training rows;
- cosine nearest-centroid routing with `top_p=4` and equal cluster weight;
- per-prompt z-score of training quality across model columns, clipped to
  `[-3, 3]` and linearly mapped to `[0, 1]`;
- empirical-Bayes cluster means with `shrinkage_k0=10`;
- per-cluster min-max quality normalization;
- global min-max normalization of train-derived mean model cost;
- direct `alpha * quality + (1-alpha) * cheapness` scoring;
- stable manifest/model-column order for ties.

The quality-bias sweep is used directly as alpha. Production-only dial
calibration, per-cluster alpha floors, provider filters, subscription subsidies,
and tool/image capability rules are excluded because they are learned from or
specific to a different production roster and are not part of the core
public-data algorithm comparison.

Two alternatives are rejected. Loading the literal v0.75 artifact cannot be
scored because its selected models lack outcomes in LLMRouterBench. Using the
production Jina embedder only for Weave would confound the policy comparison;
both policies therefore use the already pinned BGE vectors.

The metadata source is
`internal/router/cluster/artifacts/v0.75/metadata.yaml`. Trainer mechanics not
serialized there are pinned to repository revision
`b74af8941cb71e3a0bc43af9d9a68b0c727fb7cf`, specifically
`scripts/train_cluster_router.py` and `scripts/bench_walker.py`.

## Shared experimental controls

Both policies receive identical:

- accepted prompt rows and 13-model outcome matrix;
- prompt-hash 70/15/15 split with seed 42;
- BGE-small ONNX embedding vectors;
- training rows for fitting quality and cost statistics;
- held-out test rows for scoring;
- quality-bias values `0, 0.25, 0.5, 0.75, 1`;
- train-selected best-single reference;
- micro/macro aggregation and paired bootstrap procedure.

`weave_v075` is fixed from repository metadata and does not inspect validation
or test outcomes. The Codex policy remains selected on validation only. Neither
policy may use test outcomes before its selections are frozen.

## Components

`codex/src/routerlab/weave.py` owns pure Weave fitting and vector selection.
It exposes a validated immutable configuration, a fitted policy, prompt-wise
z-scoring, and `fit_weave_policy`. The module has no filesystem or network I/O.

`codex/src/routerlab/pipeline.py` fits the Weave policy from training rows during
evaluation, evaluates it beside all existing policies, and emits:

- `weave_v075` points on the shared quality-cost sweep;
- its cost saving at best-single quality;
- per-bias relative gains over best single for Weave and Codex;
- paired Codex-minus-Weave quality and cost confidence intervals;
- a deterministic digest of the fitted Weave tensors and configuration.

`codex/src/routerlab/report.py` renders a dedicated algorithm comparison table
before the generic baseline table and states the controls above explicitly.

## Result interpretation

The primary quantity is each router's held-out quality gain over the same
train-selected best single model. Their difference equals the paired
Codex-minus-Weave quality difference, but both gains are printed to make the
reference explicit.

At a particular alpha, Codex may be described as higher quality only when the
paired quality interval is above zero. A quality-cost point dominates only when
it has no lower quality and no higher cost, with one strict inequality. The
report must not claim one global winner if the frontiers cross.

## Validation

Tests must establish:

- prompt z-scoring, shrinkage, equal top-P summation, cost blending, and stable
  ties match hand-calculated examples;
- fixed v0.75 configuration values are recorded in results;
- changing every held-out test outcome cannot change the Weave policy digest or
  selections;
- the pipeline reports both relative gains and paired intervals;
- the rendered report names this as an algorithm-level comparison;
- existing Python, Rust, and Go router verification remains green.

The public aggregate result files are regenerated only after these tests pass.
