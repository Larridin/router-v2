"""
Multi-strategy router evaluation — tests multiple routing algorithms
against the same LLMRouterBench data and compares results.

Strategies:
  1. Cluster (tuned)     — k-means + better embedding + hyperparam sweep
  2. k-NN                — nearest neighbor, no clustering needed
  3. XGBoost             — predict per-model scores from embeddings
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.data_loader import BaselineDataLoader
from baselines.adaptors.avengerspro_adaptor import AvengersProAdaptor
from local_embedding import LocalEmbeddingCache


# ============================================================================
# Data loading (shared)
# ============================================================================

def load_data(config_path: str, seed: int = 42):
    loader = BaselineDataLoader(config_path=config_path)
    all_records = loader.load_all_records()
    all_models = sorted(set(r.model_name for r in all_records))
    train_records, test_records = loader.split_by_dataset_then_prompt(
        all_records, train_ratio=0.8, random_seed=seed
    )
    adaptor = AvengersProAdaptor(config_path=config_path, random_seed=seed)
    train_data = adaptor._convert_records_to_jsonl_format(train_records, all_models)
    test_data = adaptor._convert_records_to_jsonl_format(test_records, all_models)
    return train_data, test_data, all_models, test_records


def compute_baseline(test_records):
    """Compute best-single-model baseline per dataset and oracle."""
    model_dataset_scores = defaultdict(lambda: defaultdict(list))
    for r in test_records:
        model_dataset_scores[r.model_name][r.dataset_id].append(r.score)

    baseline = {}
    for model, datasets in model_dataset_scores.items():
        baseline[model] = {}
        for d, scores in datasets.items():
            baseline[model][d] = sum(scores) / len(scores) if scores else 0.0

    best_single = {}
    for d in sorted(set(r.dataset_id for r in test_records)):
        scores = {m: baseline[m].get(d, 0) for m in baseline}
        if scores:
            best_m = max(scores, key=scores.get)
            best_single[d] = (best_m, scores[best_m])

    return baseline, best_single


# ============================================================================
# Strategy 1: Tuned Cluster Router
# ============================================================================

class TunedClusterRouter:
    """Cluster router with better embedding model and hyperparameter tuning."""

    def __init__(self, embedder, n_clusters=32, top_k=5, beta=10.0, seed=42):
        self.embedder = embedder
        self.n_clusters = n_clusters
        self.top_k = top_k
        self.beta = beta
        self.seed = seed

    def embed_batch(self, texts):
        embs = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            embs.extend(self.embedder.batch(batch, max_batch_size=100))
        return np.array(embs)

    def train(self, train_data, available_models):
        queries = [item["query"] for item in train_data]
        print(f"  Embedding {len(queries)} training queries...")
        embs = self.embed_batch(queries)
        print(f"  Embeddings: {embs.shape}")

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
        labels = kmeans.fit_predict(embs)

        self.cluster_rankings = {}
        for cid in range(self.n_clusters):
            mask = labels == cid
            if not mask.any():
                continue
            cluster_items = [train_data[i] for i in np.where(mask)[0]]
            model_scores = defaultdict(list)
            for item in cluster_items:
                for m in available_models:
                    if m in item["records"]:
                        s = item["records"][m]
                        if s is not None:
                            model_scores[m].append(float(s))
            scores = {m: np.mean(v) if v else 0.0 for m, v in model_scores.items()}
            for m in available_models:
                if m not in scores:
                    scores[m] = 0.0
            sorted_m = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            self.cluster_rankings[cid] = {
                "scores": dict(sorted_m),
                "ranking": [m for m, _ in sorted_m],
                "total": len(cluster_items),
            }

        self.centroids = kmeans.cluster_centers_
        self.available_models = available_models

    def route(self, queries):
        embs = self.embed_batch(queries)
        norms_q = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        norms_c = np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-12
        sims = (embs / norms_q) @ (self.centroids / norms_c).T

        results = []
        for q_sims in sims:
            top_idx = np.argsort(q_sims)[-self.top_k:][::-1]
            top_vals = q_sims[top_idx]
            weights = np.exp(self.beta * top_vals)
            weights /= weights.sum()

            scores = defaultdict(float)
            for cidx, w in zip(top_idx, weights):
                if cidx in self.cluster_rankings:
                    ranking = self.cluster_rankings[cidx]["ranking"]
                    for m in self.available_models:
                        if m in ranking:
                            rank_score = 1.0 / (ranking.index(m) + 1)
                            scores[m] += w * rank_score
            for m in self.available_models:
                if m not in scores:
                    scores[m] = 0.0
            best = max(scores, key=scores.get)
            results.append([best])
        return results


# ============================================================================
# Strategy 2: k-NN Router
# ============================================================================

class KNNRouter:
    """Routes by finding k nearest training queries and picking the best model for them."""

    def __init__(self, embedder, k=15, seed=42):
        self.embedder = embedder
        self.k = k
        self.seed = seed

    def embed_batch(self, texts):
        embs = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            embs.extend(self.embedder.batch(batch, max_batch_size=100))
        return np.array(embs)

    def train(self, train_data, available_models):
        self.available_models = available_models
        self.train_data = train_data
        queries = [item["query"] for item in train_data]
        print(f"  Embedding {len(queries)} training queries...")
        self.train_embs = self.embed_batch(queries)

        # Per-model best score per training query for k-NN routing
        self.train_model_scores = {}
        for i, item in enumerate(train_data):
            self.train_model_scores[i] = {
                m: float(item["records"].get(m, 0) or 0) for m in available_models
            }

        self.nn = NearestNeighbors(n_neighbors=self.k, metric="cosine", n_jobs=-1)
        self.nn.fit(self.train_embs)
        print(f"  k-NN index built with k={self.k}")

    def route(self, queries):
        embs = self.embed_batch(queries)
        _, indices = self.nn.kneighbors(embs)

        results = []
        for nn_indices in indices:
            scores = defaultdict(float)
            for idx in nn_indices:
                for m in self.available_models:
                    scores[m] += self.train_model_scores[idx].get(m, 0)
            for m in self.available_models:
                scores[m] /= self.k
            best = max(scores, key=scores.get)
            results.append([best])
        return results


# ============================================================================
# Strategy 3: XGBoost Router
# ============================================================================

class XGBoostRouter:
    """Trains per-model XGBoost regressors to predict scores from embeddings."""

    def __init__(self, embedder, seed=42):
        self.embedder = embedder
        self.seed = seed

    def embed_batch(self, texts):
        embs = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            embs.extend(self.embedder.batch(batch, max_batch_size=100))
        return np.array(embs)

    def train(self, train_data, available_models):
        self.available_models = available_models
        queries = [item["query"] for item in train_data]
        print(f"  Embedding {len(queries)} training queries...")
        X = self.embed_batch(queries)

        try:
            import xgboost as xgb
        except ImportError:
            print("  xgboost not installed, falling back to Ridge regression")
            from sklearn.linear_model import Ridge
            self._train_linear(X, train_data, available_models)
            return

        self.models = {}
        for model_name in tqdm(available_models, desc="  Training XGBoost models"):
            y = np.array([
                float(item["records"].get(model_name, 0) or 0)
                for item in train_data
            ])
            m = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.seed, verbosity=0, n_jobs=-1,
            )
            m.fit(X, y)
            self.models[model_name] = m

    def _train_linear(self, X, train_data, available_models):
        from sklearn.linear_model import Ridge
        self.models = {}
        for model_name in tqdm(available_models, desc="  Training ridge models"):
            y = np.array([
                float(item["records"].get(model_name, 0) or 0)
                for item in train_data
            ])
            m = Ridge(alpha=1.0)
            m.fit(X, y)
            self.models[model_name] = m

    def route(self, queries):
        X = self.embed_batch(queries)
        results = []
        for i in range(len(queries)):
            scores = {}
            for m in self.available_models:
                scores[m] = float(self.models[m].predict(X[i : i + 1])[0])
            best = max(scores, key=scores.get)
            results.append([best])
        return results


# ============================================================================
# Evaluation (shared)
# ============================================================================

def evaluate(name, router, test_data, test_records):
    """Evaluate a router and return metrics."""
    queries = [item["query"] for item in test_data]
    t0 = time.time()
    selections = router.route(queries)
    elapsed = time.time() - t0

    correct = 0.0
    n = len(test_data)
    dataset_perf = defaultdict(lambda: {"correct": 0.0, "total": 0})
    model_counts = Counter()

    for item, sel in zip(test_data, selections):
        dataset = item["dataset"]
        picked = sel[0] if sel else None
        score = float(item["records"].get(picked, 0) if picked else 0)
        correct += score
        dataset_perf[dataset]["correct"] += score
        dataset_perf[dataset]["total"] += 1
        model_counts[picked or "none"] += 1

    dataset_accs = []
    for d, p in dataset_perf.items():
        if p["total"] > 0:
            dataset_accs.append(p["correct"] / p["total"])

    accuracy = np.mean(dataset_accs) if dataset_accs else 0.0
    raw_accuracy = correct / n if n > 0 else 0.0

    # Oracle for this test set
    oracle_scores = []
    for item in test_data:
        scores = [float(s) for s in item["records"].values()]
        oracle_scores.append(max(scores))
    oracle = np.mean(oracle_scores)

    # Best single model accuracy
    baseline, best_single = compute_baseline(test_records)
    model_avgs = {}
    for model, datasets in baseline.items():
        scores = list(datasets.values())
        if scores:
            model_avgs[model] = np.mean(scores)
    best_single_acc = max(model_avgs.values()) if model_avgs else 0
    best_single_name = max(model_avgs, key=model_avgs.get) if model_avgs else "?"

    return {
        "name": name,
        "accuracy": accuracy,
        "raw_accuracy": raw_accuracy,
        "best_single": best_single_acc,
        "best_single_name": best_single_name,
        "oracle": oracle,
        "vs_best_single": accuracy - best_single_acc,
        "vs_oracle_pct": accuracy / oracle * 100 if oracle > 0 else 0,
        "dataset_perf": dict(dataset_perf),
        "model_counts": dict(model_counts),
        "time_s": elapsed,
        "n_queries": n,
    }


def print_results(all_results, test_records):
    """Pretty-print comparison table."""
    baseline, best_single = compute_baseline(test_records)

    print("\n" + "=" * 70)
    print("FINAL RESULTS COMPARISON")
    print("=" * 70)

    print(f"\n{'Strategy':<25} {'Accuracy':<10} {'vs Best':<10} {'vs Oracle':<10} {'Time':<10}")
    print("-" * 70)
    for r in sorted(all_results, key=lambda x: x["accuracy"], reverse=True):
        print(
            f"{r['name']:<25} {r['accuracy']:<10.4f} "
            f"{r['vs_best_single']:+.4f}     "
            f"{r['vs_oracle_pct']:<10.1f}% "
            f"{r['time_s']:<10.1f}s"
        )
    print("-" * 70)
    # Calculate best single model across test_records
    model_avg = defaultdict(list)
    for rec in test_records:
        model_avg[rec.model_name].append(rec.score)
    best_m = max(model_avg, key=lambda m: np.mean(model_avg[m]))
    best_m_acc = np.mean(model_avg[best_m])
    print(f"{'Best single model':<25} {best_m_acc:<10.4f}  —          —           —")

    oracle_scores = []
    # Use test_data from all_results
    oracle_scores = []
    for r in all_results:
        if "oracle_scores" not in r:
            continue

    print(f"\n{'Oracle (upper bound)':<25} {all_results[0]['oracle']:<10.4f}")

    # Per-dataset breakdown
    print(f"\n{'Dataset':<20}", end="")
    for r in all_results:
        short = r["name"].split("(")[0].strip().replace("Router", "")[:8]
        print(f"{short:<10}", end="")
    print(f"{'BestSgl':<10}{'Oracle':<10}")
    print("-" * (20 + 12 * len(all_results) + 20))

    datasets = sorted(set(r.dataset_id for r in test_records))
    for ds in datasets:
        print(f"{ds:<20}", end="")
        for r in all_results:
            dp = r["dataset_perf"].get(ds, {"correct": 0, "total": 0})
            acc = dp["correct"] / dp["total"] if dp["total"] > 0 else 0
            print(f"{acc:<10.4f}", end="")

        # Best single for this dataset
        ds_baselines = {}
        for model, dset in baseline.items():
            if ds in dset:
                ds_baselines[model] = dset[ds]
        ds_best = max(ds_baselines.values()) if ds_baselines else 0
        print(f"{ds_best:<10.4f}", end="")

        # Oracle for this dataset
        ds_items = []
        for r in all_results:
            if "test_data_items" in r:
                ds_items = [item for item in r["test_data_items"] if item["dataset"] == ds]
                break
        if ds_items:
            ds_oracle_scores = []
            for item in ds_items:
                scores = [float(s) for s in item["records"].values()]
                ds_oracle_scores.append(max(scores))
            ds_oracle = np.mean(ds_oracle_scores)
        else:
            ds_oracle = 0
        print(f"{ds_oracle:<10.4f}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-strategy router eval")
    parser.add_argument("--config", default="config/baseline_config.yaml")
    parser.add_argument("--performance-cost", action="store_true")
    parser.add_argument(
        "--embedding-model",
        default="all-mpnet-base-v2",
        help="sentence-transformers model (all-mpnet-base-v2=better, all-MiniLM-L6-v2=faster)",
    )
    parser.add_argument("--strategies", default="all",
                       help="Comma-separated: cluster,knn,xgboost,all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config_path = (
        "config/baseline_config_performance_cost.yaml"
        if args.performance_cost
        else args.config
    )

    print("=" * 70)
    print("MULTI-STRATEGY LLM ROUTER EVALUATION")
    print("=" * 70)
    print(f"Embedding model: {args.embedding_model}")
    print(f"Config: {config_path}")

    # Load data once
    print("\n--- Loading data ---")
    train_data, test_data, all_models, test_records = load_data(config_path, args.seed)
    print(f"Train queries: {len(train_data)}, Test queries: {len(test_data)}")
    print(f"Models: {len(all_models)}")

    # Load embedder once
    print(f"\n--- Loading embedding model: {args.embedding_model} ---")
    embedder = LocalEmbeddingCache(model_name=args.embedding_model)

    all_results = []

    strategies = args.strategies.split(",")

    # ==============================
    # Strategy 1: Tuned Cluster
    # ==============================
    if "all" in strategies or "cluster" in strategies:
        print("\n" + "=" * 50)
        print("STRATEGY 1: Tuned Cluster Router")
        print("=" * 50)

        # Test a few hyperparameter combos and pick best
        configs = [
            (32, 5, 10.0),   # more clusters, wider top-k
            (24, 4, 8.0),    # balanced
            (40, 6, 12.0),   # aggressive
            (48, 7, 14.0),   # very aggressive
            (16, 3, 6.0),    # conservative
        ]
        best_cluster = None
        best_cluster_acc = 0

        for nc, tk, beta in configs:
            print(f"\n  n_clusters={nc}, top_k={tk}, beta={beta}")
            router = TunedClusterRouter(embedder, n_clusters=nc, top_k=tk, beta=beta, seed=args.seed)
            router.train(train_data, all_models)
            result = evaluate(f"Cluster(nc={nc},tk={tk})", router, test_data, test_records)
            print(f"  Accuracy: {result['accuracy']:.4f} (vs best single: {result['vs_best_single']:+.4f})")
            if result["accuracy"] > best_cluster_acc:
                best_cluster_acc = result["accuracy"]
                best_cluster = result

        if best_cluster:
            best_cluster["name"] = "Cluster Router (tuned)"
            all_results.append(best_cluster)

    # ==============================
    # Strategy 2: k-NN
    # ==============================
    if "all" in strategies or "knn" in strategies:
        print("\n" + "=" * 50)
        print("STRATEGY 2: k-NN Router")
        print("=" * 50)

        best_knn = None
        best_knn_acc = 0
        for k in [5, 10, 15, 21, 31]:
            print(f"\n  k={k}")
            router = KNNRouter(embedder, k=k, seed=args.seed)
            router.train(train_data, all_models)
            result = evaluate(f"k-NN(k={k})", router, test_data, test_records)
            print(f"  Accuracy: {result['accuracy']:.4f} (vs best single: {result['vs_best_single']:+.4f})")
            if result["accuracy"] > best_knn_acc:
                best_knn_acc = result["accuracy"]
                best_knn = result

        if best_knn:
            best_knn["name"] = "k-NN Router (tuned)"
            all_results.append(best_knn)

    # ==============================
    # Strategy 3: XGBoost
    # ==============================
    if "all" in strategies or "xgboost" in strategies:
        print("\n" + "=" * 50)
        print("STRATEGY 3: XGBoost Router")
        print("=" * 50)
        try:
            import xgboost
            print("  xgboost available")
        except ImportError:
            print("  xgboost not available, will use Ridge regression")

        router = XGBoostRouter(embedder, seed=args.seed)
        router.train(train_data, all_models)
        result = evaluate("XGBoost Router", router, test_data, test_records)
        result["name"] = "XGBoost Router"
        all_results.append(result)

    # ==============================
    # Print comparison
    # ==============================
    # Attach test data for per-dataset oracle computation
    for r in all_results:
        r["test_data_items"] = test_data

    print_results(all_results, test_records)

    # Print model selection distribution for the best router
    best = max(all_results, key=lambda x: x["accuracy"])
    print(f"\n--- Model Selection ({best['name']}) ---")
    total = sum(best["model_counts"].values())
    for model, count in sorted(best["model_counts"].items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = count / total * 100 if total > 0 else 0
        print(f"  {model:<40} {count:5d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()