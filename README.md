# router-v2

Smart LLM model router — picks the best model for every prompt. Researched, built, and evaluated from scratch using public benchmarks and open-source code.

**Three components, one repo:**

| Component | What it does |
|---|---|
| `router/` | Weave open-source router (Go, AvengersPro cluster scorer) — reference implementation |
| `claude/routerlab/` | Our bundle-based router — train once, route prompts via CLI with live pricing |
| `router-bench/` | Eval harness — compares routing algorithms on public LLMRouterBench data |

## Routers in this repo

We have **four router implementations** built during this research:

| Router | Location | Models | Pricing | Decision Rule | Status |
|---|---|---|---|---|---|
| **LLM Bundle** (ours) | `claude/routerlab/llm_bundle.py` | 13 flagship | litellm live | dollar-EV | Production-ready bundle v0.3 |
| **Routerlab** (Claude-built) | `claude/routerlab/bundle.py` | 11 RouterBench | empirical | dollar-EV | Research bundle v0.2 |
| **Weave reference** | `router/` | Anthropic/OpenAI/Gemini | catalog | alpha-blend | Reference implementation |
| **Experimental** | `qwen/` | various | — | confidence/ensemble | Research prototypes |

## Quick start

```bash
git clone https://github.com/Larridin/router-v2.git
cd router-v2
```

### 1. Route a prompt (our router)

```bash
cd claude
uv sync --frozen
uv run python -m routerlab.llm_bundle route "Write a Python function to implement quicksort" --lam 30
```

```json
{
  "model": "deepseek-v3-0324",
  "p_success": 0.678,
  "est_cost_usd": 0.000219,
  "lambda": 30.0
}
```

**Lambda controls the quality vs cost tradeoff:**
- `--lam 0` → pure quality (most expensive model)
- `--lam 30` → balanced (good quality, much cheaper)
- `--lam 5000` → cheapest model regardless of quality

### 2. Run both routers

```bash
# Our LLM bundle router (13 flagship models, litellm pricing)
uv run python -m routerlab.llm_bundle route "Write a Python function to implement quicksort" --lam 30

# The original Routerlab router (11 RouterBench models, empirical pricing)
uv run python -m routerlab.bundle route "Write a Python function to implement quicksort" --lam 30
```

```bash
cd router-bench
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt sentence-transformers xgboost lightgbm

# Download benchmark data (6.5 GB, one-time)
wget https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/main/bench-release.tar.gz
tar xzf bench-release.tar.gz --strip-components=1 -C results/bench/

# Run cluster router on coding benchmarks
python3 run_multi.py \
  --config config/coding_config.yaml \
  --embedding-model all-mpnet-base-v2 \
  --strategies cluster
```

### 3. Retrain the bundle (with new models or data)

```bash
cd claude
uv run python -m routerlab.llm_bundle save
# Saves to models/v0.4/ (auto-incrementing)
```

## How it works

```
Prompt
  → embed into a vector (MiniLM ONNX)
  → estimate quality per model (cluster64 + kNN64 + GBT ensemble)
  → estimate cost per model (litellm live pricing)
  → pick best: quality − lambda × cost
  → return model + confidence + price
```

The core algorithm (AvengersPro, DAI 2025 Best Paper):
1. Embed prompts from training data into vectors
2. Cluster similar prompts via k-means
3. Learn which model performs best per cluster
4. At test time: find nearest clusters → ensemble their recommendations → apply cost tradeoff

## What we proved

Evaluated on LLMRouterBench (public benchmark, 400K+ instances, 33 models, 21 datasets):

| Setting | Router Accuracy | vs Best Single Model | % of Oracle |
|---|---|---|---|
| 7B models, 10 coding benchmarks | **77.7%** | +4.1% better | 82.7% |
| Flagship models, 7 coding benchmarks | **69.3%** | +4.5% better | 86.9% |
| All 15 benchmarks (incl. sentiment, medical) | 71.5% | +5.2% better | 79.8% |

The cluster-based router consistently beats picking any single model blindly, and achieves ~60% cost savings at equal quality by routing to cheaper models when they're good enough.

## Strategies tested

| Strategy | Result | Notes |
|---|---|---|
| **Cluster router** | Winner (77.7%) | Naturally regularized — smooths decisions across similar prompts |
| k-NN router | Close (77.2%) | Simpler but noisier |
| Hybrid (embed + keywords) | Worse (71.4%) | Keywords added noise |
| Random Forest classifier | Failed (68.3%) | Overfits with 20 classes on 4.7K samples |
| XGBoost / LightGBM | Too slow | 94K-row training matrix |

## Architecture

```
router-v2/
├── router/                    # Weave reference router (Go, AvengersPro cluster scorer)
│   └── internal/router/cluster/artifacts/  # Pre-trained centroids & rankings (v0.75)
│
├── claude/                    # Our router & research
│   ├── routerlab/
│   │   ├── llm_bundle.py      # Bundle system: save/load/route with litellm pricing
│   │   ├── llm_data.py        # LLMRouterBench data loader
│   │   ├── baseline.py        # Weave/AvengersPro cluster router reimplementation
│   │   ├── estimators.py      # Cluster, kNN, GBT quality estimators
│   │   ├── rules.py           # Dollar-EV & alpha-blend decision rules
│   │   ├── bundle.py          # Original RouterBench bundle (11 models, v0.2)
│   │   └── embedder.py        # MiniLM ONNX embedder
│   ├── models/
│   │   └── v0.3/              # Trained bundle: 13 flagship models, 7 coding datasets
│   └── crossbench.py          # Cross-benchmark: our router vs Weave on Codex's data
│
├── router-bench/              # Eval harness
│   ├── run_multi.py           # Multi-strategy comparison harness
│   ├── local_embedding.py     # Local sentence-transformers (no API keys needed)
│   ├── config/
│   │   ├── coding_config.yaml         # Coding-only datasets (7B models)
│   │   └── coding_flagship_config.yaml # Flagship models + coding datasets
│   └── results/bench/         # LLMRouterBench data (downloaded separately)
│
├── qwen/                      # Experimental routers
│   ├── confidence_router.py   # Confidence-based routing
│   └── ensemble_router.py     # Ensemble routing experiments
│
└── codex/results/REPORT.md    # Codex vs Weave head-to-head comparison
```

## Model roster (v0.3 bundle)

13 flagship models trained on 7 coding-relevant benchmarks:

| Model | Quality (relative) | Input $/1M tok | Output $/1M tok |
|---|---|---|---|
| gpt-5 | Highest | $1.25 | $10.00 |
| claude-sonnet-4 | High | $3.00 | $15.00 |
| gemini-2.5-pro | High | $1.25 | $10.00 |
| gemini-2.5-flash | Good | $0.30 | $2.50 |
| deepseek-v3.1-terminus | Good | $0.27 | $1.00 |
| deepseek-r1-0528 | Good | $0.50 | $2.15 |
| qwen3-235b-thinking | Good | $0.18 | $0.54 |
| kimi-k2-0905 | Good | $0.60 | $3.00 |
| gpt-5-chat | Good | $1.25 | $10.00 |
| deepseek-v3-0324 | Good | $0.25 | $0.88 |
| glm-4.6 | Fair | $2.25 | $2.75 |
| intern-s1 | Fair | $0.18 | $0.54 |
| qwen3-235b | Fair | $0.18 | $0.54 |

Pricing is live from [litellm](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json).

## References

- **AvengersPro**: Zhang et al., "Beyond GPT-5: Making LLMs Cheaper and Better via Performance-Efficiency Optimized Routing", DAI 2025 Best Paper. [arXiv:2508.12631](https://arxiv.org/abs/2508.12631)
- **LLMRouterBench**: Li et al., "A Massive Benchmark and Unified Framework for LLM Routing", ACL 2026 Findings. [arXiv:2601.07206](http://arxiv.org/abs/2601.07206)
- **Weave Router**: [github.com/workweave/router](https://github.com/workweave/router)
- **AvengersPro code**: [github.com/ZhangYiqun018/AvengersPro](https://github.com/ZhangYiqun018/AvengersPro)
- **LLMRouterBench data**: [huggingface.co/datasets/NPULH/LLMRouterBench](https://huggingface.co/datasets/NPULH/LLMRouterBench)
- **Litellm pricing**: [github.com/BerriAI/litellm](https://github.com/BerriAI/litellm)

## Reports

- `claude/claude-report-aug2-v1.md` — Routerlab: what we built
- `claude/codex-report-aug2-v1.md` — Codex vs Weave head-to-head
- `router/deep-seek-report-aug2-v1.md` — Full research report: all results, strategies, findings

## License

This repository contains code from multiple sources. See individual directories for license information. The Weave router (`router/`) is licensed under Elastic License 2.0. LLMRouterBench is MIT.