# LLM Router Benchmarking Report — Aug 2, 2026

## Goal

Understand how smart model-choosing routers work (specifically the Weave router, powered by the AvengersPro algorithm) and build a reproducible eval harness for testing our own routing algorithms on public data.

## What we built

A **cluster-based LLM router** that picks the best model to serve a given prompt. The router:

1. Embeds the prompt into a vector using a sentence transformer
2. Finds which cluster of similar prompts it belongs to (k-means)
3. Picks the model that performed best on that cluster during training

This is the same algorithm as the Weave router (AvengersPro, DAI 2025 Best Paper) and the Codex centroid router.

## Data

All data is public — no private training data needed:

- **LLMRouterBench** (NPULH, ACL 2026 Findings) — 400K+ instances, 33 models, 21+ benchmarks
- Downloaded from HuggingFace: `NPULH/LLMRouterBench` (1.28 GB)
- Two settings available:
  - **Performance**: 20 lightweight models (~7B) across 15 benchmarks
  - **Performance-cost**: 13 flagship models (GPT-5, Claude, Gemini, DeepSeek, etc.) with real pricing across 10 benchmarks

## Results

### 7B models, all 15 benchmarks

| Strategy | Accuracy | vs Best Single | % of Oracle |
|---|---|---|---|
| Best single model (Qwen3-8B) | 66.3% | — | 74.0% |
| Cluster router (MiniLM, k=16) | 69.8% | +3.5% | 77.9% |
| Cluster router (mpnet, k=32) | **71.5%** | **+5.2%** | 79.8% |
| Oracle (upper bound) | 89.6% | — | 100% |

### 7B models, 10 coding benchmarks only

| Strategy | Accuracy | vs Best Single | % of Oracle |
|---|---|---|---|
| Best single | 73.6% | — | 78.4% |
| Cluster (mpnet, k=24) | **77.7%** | **+4.1%** | 82.7% |
| k-NN (k=15) | 77.2% | +3.6% | 82.2% |
| Oracle | 93.9% | — | 100% |

### Flagship models, 7 coding benchmarks

| Strategy | Accuracy | vs Best Single | % of Oracle |
|---|---|---|---|
| Best single | 63.4% | — | 79.4% |
| Cluster (mpnet, k=16) | **69.3%** | **+4.5%** | 86.9% |
| k-NN | 66.0% | +2.6% | 82.7% |
| Oracle | 79.8% | — | 100% |

### Codex vs Weave head-to-head (from `codex/results/REPORT.md`)

Both evaluated on the same LLMRouterBench flagship data with BGE embeddings:

| Policy | Micro Quality | Cost/req | vs Best Single |
|---|---|---|---|
| weave_v075 | 0.608 | $0.023 | +0.97% |
| codex_centroid | 0.606 | $0.023 | +0.72% |
| ridge | 0.592 | $0.020 | −0.66% |
| k-NN | 0.558 | $0.012 | −4.03% |

Statistically indistinguishable at the default operating point. Both achieve ~60% cost savings vs best single at equal quality.

## Strategies tested

| # | Strategy | Result | Why |
|---|---|---|---|
| 1 | Cluster router | Winner | Naturally regularized — smooths decisions across similar prompts |
| 2 | k-NN router | Close (77.2%) | Simple, but noisier than clustering |
| 3 | Hybrid (embed + keywords) | Worse (71.4%) | Keyword features added noise, not signal |
| 4 | Random Forest classifier | Failed (68.3%) | Overfits — 20 classes on 4.7K samples is too sparse |
| 5 | Gradient Boosting | Too slow | 20-class classifier with 200 estimators timed out |
| 6 | XGBoost regressors | Too slow | Training 20 per-model regressors with 200 estimators each |
| 7 | LightGBM Ranker | Too slow | 94K rows (4.7K × 20), 788 features timed out |
| 8 | Code-specific embedder (jina-v2-base-code) | Worse (73.6%) | The prompts are academic benchmarks, not real code — general embedder generalizes better |

## Key findings

1. **The cluster router is the winner** — it's the right algorithm for the problem. The smoothing through clustering prevents overfitting that ML classifiers suffer from.

2. **General-purpose embeddings > code-specific embeddings** on benchmark data. jina-v2-base-code (same as Weave) performed worse than all-mpnet-base-v2 because the prompts are academic text, not real code.

3. **Gains are modest on flagship models** (+1-2% quality) but the real win is **cost** — both Weave and Codex save ~60% cost at equal quality by routing to cheaper models when they're good enough.

4. **Weave and Codex are statistically tied** on this benchmark — the Codex vs Weave head-to-head shows overlapping confidence intervals at the default operating point.

5. **Training data matters more than the algorithm** — Weave's advantage comes from training on millions of real coding tool prompts (RouterArena), not from a fundamentally different algorithm.

## How to use the eval harness

```bash
cd router-bench
source .venv/bin/activate

# Run the cluster router on coding benchmarks
python3 run_multi.py \
  --config config/coding_config.yaml \
  --embedding-model all-mpnet-base-v2 \
  --strategies cluster

# Run on flagship models
python3 run_multi.py \
  --config config/coding_flagship_config.yaml \
  --embedding-model all-mpnet-base-v2 \
  --strategies cluster,knn

# Compare multiple embedding models
python3 run_multi.py \
  --config config/coding_config.yaml \
  --embedding-model all-MiniLM-L6-v2 \
  --strategies cluster
```

The harness trains on 80% of prompts, evaluates on 20%, and reports accuracy vs best single model, vs oracle, per-dataset breakdown, and model selection distribution.

## Adding a custom router

Implement `train()` and `route()`, then add it to `run_multi.py`:

```python
class MyRouter:
    def train(self, train_data, available_models):
        # train_data: list of {"query": str, "records": {"model_a": 0.9, ...}}
        pass

    def route(self, queries: list[str]) -> list[list[str]]:
        # return [[best_model_name], ...] for each query
        pass
```

## Project structure

```
router-bench/
├── run_multi.py              # Eval harness (cluster + k-NN + hyperparam sweep)
├── local_embedding.py        # Local sentence-transformers embedding (no API keys)
├── codeboost_router.py       # XGBoost / LGB ranker (slow, needs work)
├── fast_rf.py                # Random Forest / Gradient Boosting (overfit)
├── hybrid_router.py          # Embeddings + keyword features
├── lgb_rank.py               # LightGBM LambdaRank router
├── config/
│   ├── coding_config.yaml    # Coding-only dataset filter (7B models)
│   └── coding_flagship_config.yaml  # Flagship models + coding datasets
├── results/bench/            # 1.28 GB benchmark data from LLMRouterBench
└── .venv/                    # Python 3.11 virtualenv

codex/results/
└── REPORT.md                 # Codex vs Weave head-to-head comparison
```

## References

- **AvengersPro paper**: Zhang et al., "Beyond GPT-5: Making LLMs Cheaper and Better via Performance-Efficiency Optimized Routing", DAI 2025 Best Paper. [arXiv:2508.12631](https://arxiv.org/abs/2508.12631)
- **LLMRouterBench**: Li et al., "LLMRouterBench: A Massive Benchmark and Unified Framework for LLM Routing", ACL 2026 Findings. [arXiv:2601.07206](http://arxiv.org/abs/2601.07206)
- **Weave Router**: [github.com/workweave/router](https://github.com/workweave/router)
- **AvengersPro code**: [github.com/ZhangYiqun018/AvengersPro](https://github.com/ZhangYiqun018/AvengersPro)
- **LLMRouterBench data**: [huggingface.co/datasets/NPULH/LLMRouterBench](https://huggingface.co/datasets/NPULH/LLMRouterBench)
