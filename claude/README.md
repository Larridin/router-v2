# routerlab — core routing algorithm eval + a new router

**Status: exploratory prototype** (research harness, not production code).

## Goal

Isolate the *pure core routing* problem — "given a prompt, pick the best model" —
from everything else a production router does (session pins, prompt-cache EV,
handover, fallback). Then:

1. Reimplement the **existing** router's core algorithm as a baseline.
2. Build a **new** algorithm.
3. Run both through **identical evals** on a public label matrix and compare.

## The two routers

| | quality estimate | everything else |
|---|---|---|
| `baseline.ClusterRouter` | AvengersPro (arXiv 2508.12631), the algorithm behind the Weave router's cluster scorer: k-means k=16 over prompt embeddings, per-cluster shrunk quality means, top-p=4 nearest clusters summed | identical |
| `knn_router.KNNRouter` | kNN-local: similarity-weighted quality of the K=64 nearest *training prompts*, shrunk toward the global mean by effective sample size | identical |

Baseline hyperparameters (k=16, top_p=4, shrinkage k0=10, per-prompt z-scoring,
seed 42) are taken from the production bundle's `metadata.yaml` (v0.75).
Both routers share the same embedder, cost axis, alpha blend
(`alpha·quality + (1−alpha)·cheapness`), and candidate roster — the *only*
difference is cluster-quantized vs local quality estimation.

## Data & evals

- **RouterBench 0-shot** (`withmartian/routerbench`, Apache 2.0): 36.5k prompts,
  11 models, per-(prompt, model) graded quality **and measured cost**.
- Deduped, stratified 70/30 prompt-level split by benchmark source, seed 42.
- Eval = route each held-out prompt, look up what the pick earned:
  mean quality (bootstrap CIs), mean realized $ per prompt, oracle-gap-recovered,
  cost–quality frontier swept over alpha ∈ [0,1], quality-at-matched-cost table.
- References on every plot: per-prompt **oracle**, best single model, cheapest
  model, and each single model as a scatter point.

## Key assumptions

1. **Algorithm-level comparison, not artifact-level.** The literal production
   artifact routes today's roster (opus-4-8, gpt-5.6, …) which has no public
   per-prompt labels; RouterBench's roster is 2024-era. So we compare the
   *algorithms* on a common public roster. Verdicts transfer to the extent the
   algorithms, not the specific models, drive the result.
2. RouterBench costs are treated as the deployment's cost axis (mean per-model
   train cost ≈ the production `model_axes.json` analog); realized eval cost
   uses the dataset's per-prompt measured cost of the chosen model.
3. Embedder is MiniLM-L6-v2 ONNX (mean-pool, L2, tail-1024-char truncation to
   mirror the production `TailTruncate`), not the production Jina model —
   shared by both routers, so the comparison stays fair.

## Run

```bash
uv run pytest                    # core math unit tests
uv run python -m routerlab.run   # full pipeline -> results/
```

Outputs: `results/metrics.json`, `results/frontier.png`, `results/report.md`.

## Layout

```
routerlab/
├── dataset.py     # RouterBench load, dedupe, stratified split
├── embedder.py    # MiniLM ONNX (TF-IDF+SVD offline fallback), disk cache
├── core.py        # z-score, shrinkage, min-max, blend, argmax (unit-tested)
├── baseline.py    # existing algorithm reimplementation
├── knn_router.py  # new algorithm
├── evals.py       # references, alpha sweep, bootstrap CIs, quality@cost
└── run.py         # end-to-end orchestration + plot + report
```

## Findings (see results/robustness.md)

1. **At the production recipe's k=16, kNN dominates the mid-cost frontier**
   (+7.6 quality pts @ $2/1k) and the cluster router has a dial dead zone.
2. **But the gap is resolution, not algorithm**: cluster k=256 matches the best
   kNN config everywhere (0.756 vs 0.750 @ $2/1k, peak 0.784 vs 0.784) and the
   dead zone disappears. Both are neighborhood quality estimators; with enough
   clusters they converge. Cluster count should scale with labeled-corpus size
   (~100+ prompts/cluster: 25.5k train → k=256 works; the production bundle's
   1,775 prompts → k=16 is the same ratio).
3. **Leave-one-family-out: neither router transfers.** On an unseen task family
   both collapse to ≈best-single-model quality (cluster exactly; kNN 0.3–2.3
   pts below it). Routing wins come from having labeled data in the prompt's
   family; estimator sophistication doesn't fix out-of-distribution. Corpus
   coverage > algorithm.

## Hypothesis round (see results/hypotheses.md)

Three attempts to beat the champion (cluster k=256 × alpha):
**H1** learned discriminator (per-model GBT on embedding + text features),
**H2** dollar-EV decision rule (calibrated P(success) − λ·per-prompt-cost),
**H3** calibrated isotonic ensemble of cluster256/kNN/GBT.

Verdict: **the decision rule was the win, not the estimators.**
`ensemble × dollar_ev` beats the champion **+3.2 pts at $1/1k** (0.751 vs
0.720, ~4× CI) and +0.9 at $2/1k; every estimator improved mid-band under the
EV rule while every alpha-rule variant lost. No hypothesis moved the peak
(0.779 vs champion 0.784): the quality ceiling is estimator-saturated — H1's
extra capacity *lowered* it. The deployable envelope (best rule per budget)
weakly dominates the champion at 4/5 budgets and concedes ~0.4 pts at peak.

## H4: cheap LLM as the routing predictor (results/h4.json)

haiku (via OpenRouter) rates each prompt — difficulty 0-10, domain,
needs-reasoning — and an 8-feature head routes on that. Controls: an
embedding-only head on the SAME 3k training rows, and a hybrid.

- **Sample efficiency win**: at 3k labels, LLM features beat the embedding
  control decisively (peak 0.758 vs 0.726; q@$2 0.754 vs 0.722). The hybrid
  was worse than LLM-only (0.740) — 392 dims overfit 3k rows.
- **Absolute loss**: still below the 25.5k-embedding champion (0.758 vs 0.784,
  0/5 budget wins) — data volume beats per-label signal quality.
- **Latency kills it for the hot path**: measured p50 966ms / p95 1781ms per
  routing call vs ~10ms for the on-box embedder, plus ~$0.60/1k routing
  overhead. Right role for LLM judgment: OFFLINE labeling (bootstrap quality
  tables on real traffic cheaply) or async session scoring off the hot path —
  not per-request prediction.

## Next steps (deliberately out of scope here)

- Tier-3 labels: replay prompts from our own traffic across a current roster,
  LLM-judge pairwise, rerun the same evals on that matrix.
- Adapter for the live router's `POST /v1/route` to eval the actual artifact
  once a current-roster label matrix exists.
- Session-aware evals (pins, cache EV) — the other 80% of the production system.
