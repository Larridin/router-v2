"""
Standalone runner for training and evaluating cluster-based LLM routers
using LLMRouterBench data and local sentence-transformers embeddings.

Usage:
    python run_eval.py --config config/baseline_config.yaml [--performance-cost]
    
This script:
1. Converts LLMRouterBench data to AvengersPro format
2. Trains a k-means cluster router with local embeddings
3. Evaluates on the test set
4. Prints results (accuracy, cost analysis, baseline comparison)
"""

import sys
import os
import json
import argparse
from pathlib import Path
from loguru import logger

# Reduce log verbosity for data loading
logger.remove()
logger.add(sys.stderr, level="WARNING")

# Add the project root to path so we can import from baselines
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from tqdm import tqdm

from baselines.data_loader import BaselineDataLoader
from baselines.adaptors.avengerspro_adaptor import AvengersProAdaptor
from local_embedding import LocalEmbeddingCache


class LocalClusterRouter:
    """Cluster-based model router using local embeddings."""

    def __init__(
        self,
        embedder: LocalEmbeddingCache,
        n_clusters: int = 16,
        top_k: int = 4,
        beta: float = 9.0,
        seed: int = 42,
    ):
        self.embedder = embedder
        self.n_clusters = n_clusters
        self.top_k = top_k
        self.beta = beta
        self.seed = seed

    def train(self, train_data: list[dict], available_models: list[str]):
        """Train the cluster router on training data."""
        queries = [item["query"] for item in train_data]

        print(f"\nGenerating embeddings for {len(queries)} training queries...")
        embeddings = []
        batch_size = 100
        for i in tqdm(range(0, len(queries), batch_size), desc="Embedding"):
            batch = queries[i : i + batch_size]
            batch_embs = self.embedder.batch(batch, max_batch_size=batch_size)
            embeddings.extend(batch_embs)

        embeddings = np.array(embeddings)
        print(f"Embeddings matrix: {embeddings.shape}")

        # K-means clustering
        print(f"K-means clustering with k={self.n_clusters}...")
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.seed,
            n_init=10,
        )
        cluster_labels = kmeans.fit_predict(embeddings)

        # Per-cluster rankings
        cluster_data = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            cluster_data[label].append(train_data[i])

        self.cluster_rankings = {}
        for cluster_id, records in cluster_data.items():
            model_scores = defaultdict(list)
            for record in records:
                for model, score in record["records"].items():
                    if model in available_models and score is not None:
                        s = float(score)
                        model_scores[model].append(s)

            scores = {
                m: np.mean(v) if v else 0.0
                for m, v in model_scores.items()
            }
            for m in available_models:
                if m not in scores:
                    scores[m] = 0.0

            sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            self.cluster_rankings[cluster_id] = {
                "total": len(records),
                "scores": dict(sorted_models),
                "ranking": [m for m, _ in sorted_models],
            }

        self.kmeans = kmeans
        self.centroids = kmeans.cluster_centers_
        self.available_models = available_models

        print(f"Trained {len(self.cluster_rankings)} clusters")
        for cid in sorted(self.cluster_rankings.keys()):
            info = self.cluster_rankings[cid]
            print(
                f"  Cluster {cid}: {info['total']} samples, "
                f"top model: {info['ranking'][0]} ({info['scores'][info['ranking'][0]]:.4f})"
            )

    def route(self, queries: list[str]) -> list[list[str]]:
        """Route queries to best models."""
        embs = self.embedder.batch(queries, max_batch_size=100)
        embs = np.array(embs)
        # L2 normalize for cosine similarity
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

        centroids_norm = self.centroids / (
            np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-12
        )
        distances = 1 - embs @ centroids_norm.T

        results = []
        for query_dists in distances:
            top_indices = np.argsort(query_dists)[: self.top_k]
            top_dists = query_dists[top_indices]

            logits = -self.beta * top_dists
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()

            expert_scores = defaultdict(float)
            for cidx, prob in zip(top_indices, probs):
                if cidx not in self.cluster_rankings:
                    continue
                ranking = self.cluster_rankings[cidx]["ranking"]
                for model in self.available_models:
                    if model in ranking:
                        rank_score = 1.0 / (ranking.index(model) + 1)
                        expert_scores[model] += prob * rank_score

            for m in self.available_models:
                if m not in expert_scores:
                    expert_scores[m] = 0.0

            sorted_models = sorted(expert_scores.items(), key=lambda x: x[1], reverse=True)
            results.append([m for m, _ in sorted_models[:1]])

        return results

    def evaluate(self, test_data: list[dict]) -> dict:
        """Evaluate routing on test data."""
        queries = [item["query"] for item in test_data]

        print(f"\nRouting {len(queries)} test queries...")
        routing_results = self.route(queries)

        correct_routes = 0
        dataset_perf = defaultdict(lambda: {"correct": 0.0, "total": 0})
        model_selection = Counter()
        per_dataset = defaultdict(lambda: defaultdict(list))

        for item, selected in zip(test_data, routing_results):
            dataset = item["dataset"]
            selected_model = selected[0] if selected else None

            max_score = 0.0
            if selected_model and selected_model in item["records"]:
                max_score = float(item["records"][selected_model])

            correct_routes += max_score
            dataset_perf[dataset]["correct"] += max_score
            dataset_perf[dataset]["total"] += 1
            model_selection[selected_model or "none"] += 1
            per_dataset[dataset][selected_model or "none"].append(max_score)

        dataset_accuracies = []
        for d, p in dataset_perf.items():
            if p["total"] > 0:
                dataset_accuracies.append(p["correct"] / p["total"])

        accuracy = np.mean(dataset_accuracies) if dataset_accuracies else 0.0

        return {
            "accuracy": accuracy,
            "correct_routes": correct_routes,
            "total_queries": len(test_data),
            "dataset_performance": dict(dataset_perf),
            "model_selection_stats": dict(model_selection),
            "per_dataset": {d: dict(m) for d, m in per_dataset.items()},
        }


def load_baseline_scores_for_comparison(
    loader: BaselineDataLoader, test_records: list
) -> dict[str, dict[str, float]]:
    """Compute best single model per dataset from test records."""
    model_dataset_scores = defaultdict(lambda: defaultdict(list))
    for r in test_records:
        model_dataset_scores[r.model_name][r.dataset_id].append(r.score)

    baseline = {}
    for model, datasets in model_dataset_scores.items():
        baseline[model] = {}
        for d, scores in datasets.items():
            baseline[model][d] = sum(scores) / len(scores) if scores else 0.0
    return baseline


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate cluster router")
    parser.add_argument(
        "--config",
        default="config/baseline_config.yaml",
        help="Baseline config YAML",
    )
    parser.add_argument(
        "--performance-cost",
        action="store_true",
        help="Use performance-cost config (13 flagship models with pricing)",
    )
    parser.add_argument("--n-clusters", type=int, default=16, help="Number of clusters")
    parser.add_argument("--top-k", type=int, default=4, help="Top-K clusters for routing")
    parser.add_argument("--beta", type=float, default=9.0, help="Temperature for cluster weighting")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model name",
    )
    args = parser.parse_args()

    config_path = (
        "config/baseline_config_performance_cost.yaml"
        if args.performance_cost
        else args.config
    )

    print("=" * 60)
    print("LLM Router Evaluation Pipeline")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Clusters: {args.n_clusters}, Top-K: {args.top_k}, Beta: {args.beta}")

    # Step 1: Load and convert data
    print("\n--- Step 1: Loading and converting data ---")
    loader = BaselineDataLoader(config_path=config_path)
    all_records = loader.load_all_records()
    print(f"Loaded {len(all_records)} total records")

    # Get unique models
    all_models = sorted(set(r.model_name for r in all_records))
    print(f"Found {len(all_models)} models: {all_models}")

    # Split train/test
    train_records, test_records = loader.split_by_dataset_then_prompt(
        all_records, train_ratio=0.8, random_seed=args.seed
    )
    print(f"Train: {len(train_records)}, Test: {len(test_records)}")

    # Convert to AvengersPro format (per-query JSONL)
    adaptor = AvengersProAdaptor(config_path=config_path, random_seed=args.seed)
    train_data = adaptor._convert_records_to_jsonl_format(train_records, all_models)
    test_data = adaptor._convert_records_to_jsonl_format(test_records, all_models)
    print(f"Train queries: {len(train_data)}, Test queries: {len(test_data)}")

    # Step 2: Initialize local embedder
    print(f"\n--- Step 2: Loading embedding model: {args.embedding_model} ---")
    embedder = LocalEmbeddingCache(model_name=args.embedding_model)

    # Step 3: Train cluster router
    print("\n--- Step 3: Training cluster router ---")
    router = LocalClusterRouter(
        embedder=embedder,
        n_clusters=args.n_clusters,
        top_k=args.top_k,
        beta=args.beta,
        seed=args.seed,
    )
    router.train(train_data, all_models)

    # Step 4: Evaluate
    print("\n--- Step 4: Evaluating ---")
    results = router.evaluate(test_data)
    baseline = load_baseline_scores_for_comparison(loader, test_records)

    # Step 5: Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy (avg across datasets): {results['accuracy']:.4f}")

    # Best single model baseline
    model_avg = {}
    for model, datasets in baseline.items():
        scores = list(datasets.values())
        model_avg[model] = np.mean(scores) if scores else 0.0
    best_single = max(model_avg.items(), key=lambda x: x[1])
    print(f"Best Single Model: {best_single[0]} ({best_single[1]:.4f})")
    print(f"Router vs Best Single: {results['accuracy'] - best_single[1]:+.4f}")

    # Oracle (always pick the model with highest score per query)
    oracle_scores = []
    for item in test_data:
        scores = {m: float(s) for m, s in item["records"].items()}
        oracle_scores.append(max(scores.values()))
    oracle_acc = np.mean(oracle_scores)
    print(f"Oracle (upper bound): {oracle_acc:.4f}")
    print(f"Gap to Oracle: {oracle_acc - results['accuracy']:.4f}")
    print(f"Router / Oracle: {results['accuracy'] / oracle_acc * 100:.1f}%")

    print("\nPer-Dataset Performance:")
    print(f"{'Dataset':<20} {'Router':<10} {'Best Single':<14} {'Oracle':<10} {'# Queries':<10}")
    print("-" * 70)
    for dataset in sorted(results["dataset_performance"].keys()):
        dp = results["dataset_performance"][dataset]
        router_acc = dp["correct"] / dp["total"] if dp["total"] > 0 else 0.0

        # Best single for this dataset
        ds_baselines = {}
        for model, datasets in baseline.items():
            if dataset in datasets:
                ds_baselines[model] = datasets[dataset]
        best_ds = max(ds_baselines.values()) if ds_baselines else 0.0

        # Oracle for this dataset
        ds_items = [item for item in test_data if item["dataset"] == dataset]
        ds_oracle = 0.0
        if ds_items:
            ds_scores = []
            for item in ds_items:
                scores = {m: float(s) for m, s in item["records"].items()}
                ds_scores.append(max(scores.values()))
            ds_oracle = np.mean(ds_scores)

        print(
            f"{dataset:<20} {router_acc:<10.4f} {best_ds:<14.4f} {ds_oracle:<10.4f} {dp['total']:<10}"
        )

    print("\nModel Selection Distribution:")
    total = sum(results["model_selection_stats"].values())
    sorted_models = sorted(
        results["model_selection_stats"].items(), key=lambda x: x[1], reverse=True
    )
    for model, count in sorted_models[:10]:
        pct = count / total * 100 if total > 0 else 0
        print(f"  {model:<35} {count:5d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()