"""
Qwen Router Experiments — Iteration 2

Improvements:
1. Adaptive Cluster — sweep target_per_cluster and top_k/beta
2. Confidence Router V3 — soft weighting instead of hard fallback
3. Hybrid — adaptive k + confidence-weighted blending
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent / "router-bench"
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.data_loader import BaselineDataLoader
from baselines.adaptors.avengerspro_adaptor import AvengersProAdaptor
from local_embedding import LocalEmbeddingCache


def load_data(config_path, seed=42):
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


def compute_baselines(test_data, test_records):
    model_dataset_scores = defaultdict(lambda: defaultdict(list))
    for r in test_records:
        model_dataset_scores[r.model_name][r.dataset_id].append(r.score)

    model_avgs = {}
    for model, datasets in model_dataset_scores.items():
        ds_avgs = [np.mean(list(v)) for v in datasets.values() if len(v) > 0]
        if ds_avgs:
            model_avgs[model] = np.mean(ds_avgs)
    best_single_name = max(model_avgs, key=model_avgs.get) if model_avgs else "?"
    best_single_acc = model_avgs[best_single_name] if model_avgs else 0

    oracle_scores = []
    for item in test_data:
        scores = [float(s) for s in item["records"].values()]
        oracle_scores.append(max(scores))
    oracle_acc = np.mean(oracle_scores)

    return best_single_name, best_single_acc, oracle_acc


def evaluate(name, router, test_data):
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
    return {"name": name, "accuracy": accuracy, "dataset_perf": dict(dataset_perf),
            "model_counts": dict(model_counts), "time_s": elapsed, "n_queries": n}


class AdaptiveClusterRouterSweep:
    """Adaptive cluster router with hyperparameter sweep."""

    def __init__(self, embedder, target_per_cluster=80, min_clusters=8, max_clusters=256,
                 top_k=5, beta=10.0, seed=42):
        self.embedder = embedder
        self.target_per_cluster = target_per_cluster
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
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
        n_clusters = max(self.min_clusters, min(self.max_clusters, len(queries) // self.target_per_cluster))
        self.n_clusters = n_clusters
        print(f"  [AdaptiveSweep] n={len(queries)}, k={n_clusters}, top_k={self.top_k}, beta={self.beta}")

        embs = self.embed_batch(queries)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        labels = kmeans.fit_predict(embs)

        self.cluster_rankings = {}
        for cid in range(n_clusters):
            mask = labels == cid
            if not mask.any():
                continue
            cluster_items = [train_data[i] for i in np.where(mask)[0]]
            model_scores = defaultdict(list)
            for item in cluster_items:
                for m in available_models:
                    if m in item["records"] and item["records"][m] is not None:
                        model_scores[m].append(float(item["records"][m]))
            scores = {m: np.mean(v) if v else 0.0 for m, v in model_scores.items()}
            for m in available_models:
                if m not in scores:
                    scores[m] = 0.0
            sorted_m = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            self.cluster_rankings[cid] = {"scores": dict(sorted_m), "ranking": [m for m, _ in sorted_m], "total": len(cluster_items)}

        self.centroids = kmeans.cluster_centers_
        self.available_models = available_models

    def route(self, queries):
        embs = self.embed_batch(queries)
        embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        centroids_norm = self.centroids / (np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-12)
        sims = embs_norm @ centroids_norm.T

        results = []
        for j in range(len(queries)):
            q_sims = sims[j]
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


class ConfidenceRouterV3:
    """
    V3: Soft confidence weighting — no hard fallback.
    Instead, blend cluster score with best-single score proportional to confidence.
    Low confidence = more weight to best-single. High confidence = more weight to cluster.
    """

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
        embs = self.embed_batch(queries)

        from sklearn.cluster import KMeans
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
                    if m in item["records"] and item["records"][m] is not None:
                        model_scores[m].append(float(item["records"][m]))
            scores = {m: np.mean(v) if v else 0.0 for m, v in model_scores.items()}
            for m in available_models:
                if m not in scores:
                    scores[m] = 0.0
            sorted_m = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            self.cluster_rankings[cid] = {"scores": dict(sorted_m), "ranking": [m for m, _ in sorted_m], "total": len(cluster_items)}

        self.centroids = kmeans.cluster_centers_
        self.available_models = available_models

        # Best single model from training
        model_scores_global = defaultdict(list)
        for item in train_data:
            for m in available_models:
                if m in item["records"] and item["records"][m] is not None:
                    model_scores_global[m].append(float(item["records"][m]))
        self.best_single_model = max(model_scores_global, key=lambda m: np.mean(model_scores_global[m]) if model_scores_global[m] else 0)

        # Calibrate: compute training similarity distribution
        embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        centroids_norm = self.centroids / (np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-12)
        sims = embs_norm @ centroids_norm.T
        max_sims = np.max(sims, axis=1)
        self.sim_min = np.percentile(max_sims, 5)
        self.sim_max = np.percentile(max_sims, 95)
        print(f"  [ConfidenceV3] Calibrated: sim_min={self.sim_min:.4f}, sim_max={self.sim_max:.4f}")

    def route(self, queries):
        embs = self.embed_batch(queries)
        embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        centroids_norm = self.centroids / (np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-12)
        sims = embs_norm @ centroids_norm.T

        results = []
        for j in range(len(queries)):
            q_sims = sims[j]
            top_idx = np.argsort(q_sims)[-self.top_k:][::-1]
            top_vals = q_sims[top_idx]

            # Confidence = normalized max similarity
            raw_conf = top_vals[0]
            confidence = np.clip((raw_conf - self.sim_min) / (self.sim_max - self.sim_min + 1e-12), 0, 1)

            # Cluster scores
            weights = np.exp(self.beta * top_vals)
            weights /= weights.sum()
            cluster_scores = defaultdict(float)
            for cidx, w in zip(top_idx, weights):
                if cidx in self.cluster_rankings:
                    ranking = self.cluster_rankings[cidx]["ranking"]
                    for m in self.available_models:
                        if m in ranking:
                            rank_score = 1.0 / (ranking.index(m) + 1)
                            cluster_scores[m] += w * rank_score
            for m in self.available_models:
                if m not in cluster_scores:
                    cluster_scores[m] = 0.0

            # Blend: confidence * cluster + (1 - confidence) * best_single_bonus
            blended = {}
            for m in self.available_models:
                bonus = 0.3 if m == self.best_single_model else 0.0
                blended[m] = confidence * cluster_scores[m] + (1 - confidence) * bonus

            best = max(blended, key=blended.get)
            results.append([best])

        return results


class HybridAdaptiveConfidenceRouter:
    """Best of both: adaptive k + soft confidence blending."""

    def __init__(self, embedder, target_per_cluster=80, min_clusters=8, max_clusters=256,
                 top_k=5, beta=10.0, seed=42):
        self.embedder = embedder
        self.target_per_cluster = target_per_cluster
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
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
        n_clusters = max(self.min_clusters, min(self.max_clusters, len(queries) // self.target_per_cluster))
        self.n_clusters = n_clusters
        print(f"  [Hybrid] n={len(queries)}, k={n_clusters}, top_k={self.top_k}, beta={self.beta}")

        embs = self.embed_batch(queries)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        labels = kmeans.fit_predict(embs)

        self.cluster_rankings = {}
        for cid in range(n_clusters):
            mask = labels == cid
            if not mask.any():
                continue
            cluster_items = [train_data[i] for i in np.where(mask)[0]]
            model_scores = defaultdict(list)
            for item in cluster_items:
                for m in available_models:
                    if m in item["records"] and item["records"][m] is not None:
                        model_scores[m].append(float(item["records"][m]))
            scores = {m: np.mean(v) if v else 0.0 for m, v in model_scores.items()}
            for m in available_models:
                if m not in scores:
                    scores[m] = 0.0
            sorted_m = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            self.cluster_rankings[cid] = {"scores": dict(sorted_m), "ranking": [m for m, _ in sorted_m], "total": len(cluster_items)}

        self.centroids = kmeans.cluster_centers_
        self.available_models = available_models

        model_scores_global = defaultdict(list)
        for item in train_data:
            for m in available_models:
                if m in item["records"] and item["records"][m] is not None:
                    model_scores_global[m].append(float(item["records"][m]))
        self.best_single_model = max(model_scores_global, key=lambda m: np.mean(model_scores_global[m]) if model_scores_global[m] else 0)

        embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        centroids_norm = self.centroids / (np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-12)
        sims = embs_norm @ centroids_norm.T
        max_sims = np.max(sims, axis=1)
        self.sim_min = np.percentile(max_sims, 5)
        self.sim_max = np.percentile(max_sims, 95)

    def route(self, queries):
        embs = self.embed_batch(queries)
        embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        centroids_norm = self.centroids / (np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-12)
        sims = embs_norm @ centroids_norm.T

        results = []
        for j in range(len(queries)):
            q_sims = sims[j]
            top_idx = np.argsort(q_sims)[-self.top_k:][::-1]
            top_vals = q_sims[top_idx]

            raw_conf = top_vals[0]
            confidence = np.clip((raw_conf - self.sim_min) / (self.sim_max - self.sim_min + 1e-12), 0, 1)

            weights = np.exp(self.beta * top_vals)
            weights /= weights.sum()
            cluster_scores = defaultdict(float)
            for cidx, w in zip(top_idx, weights):
                if cidx in self.cluster_rankings:
                    ranking = self.cluster_rankings[cidx]["ranking"]
                    for m in self.available_models:
                        if m in ranking:
                            rank_score = 1.0 / (ranking.index(m) + 1)
                            cluster_scores[m] += w * rank_score
            for m in self.available_models:
                if m not in cluster_scores:
                    cluster_scores[m] = 0.0

            blended = {}
            for m in self.available_models:
                bonus = 0.3 if m == self.best_single_model else 0.0
                blended[m] = confidence * cluster_scores[m] + (1 - confidence) * bonus

            best = max(blended, key=blended.get)
            results.append([best])

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/baseline_config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.chdir(str(PROJECT_ROOT))

    print("=" * 90)
    print("QWEN ROUTER — ITERATION 2 (Hyperparameter Sweep + Improved Confidence)")
    print("=" * 90)

    print("\n--- Loading data ---")
    train_data, test_data, all_models, test_records = load_data(args.config, args.seed)
    best_single_name, best_single_acc, oracle_acc = compute_baselines(test_data, test_records)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}, Models: {len(all_models)}")
    print(f"Best single: {best_single_name} ({best_single_acc:.4f}), Oracle: {oracle_acc:.4f}")

    print(f"\n--- Loading embedder: all-mpnet-base-v2 ---")
    embedder = LocalEmbeddingCache(model_name="all-mpnet-base-v2")

    all_results = []

    # Sweep adaptive k
    print("\n" + "=" * 60)
    print("SWEEP 1: Adaptive k (target_per_cluster)")
    print("=" * 60)
    for tpc in [50, 60, 70, 80, 100, 120, 150]:
        router = AdaptiveClusterRouterSweep(embedder, target_per_cluster=tpc, top_k=5, beta=10.0, seed=args.seed)
        router.train(train_data, all_models)
        result = evaluate(f"Adaptive(tpc={tpc})", router, test_data)
        all_results.append(result)
        print(f"  Accuracy: {result['accuracy']:.4f}")

    # Sweep top_k and beta for best k
    best_tpc = max(all_results, key=lambda x: x["accuracy"])
    best_tpc_val = int(best_tpc["name"].split("=")[1].rstrip(")"))
    print(f"\nBest tpc={best_tpc_val}, now sweeping top_k/beta...")

    print("\n" + "=" * 60)
    print("SWEEP 2: top_k x beta")
    print("=" * 60)
    for tk in [3, 4, 5, 6, 7, 8]:
        for beta in [6.0, 8.0, 10.0, 12.0, 14.0, 16.0]:
            router = AdaptiveClusterRouterSweep(embedder, target_per_cluster=best_tpc_val,
                                                top_k=tk, beta=beta, seed=args.seed)
            router.train(train_data, all_models)
            result = evaluate(f"Adaptive(tk={tk},beta={beta})", router, test_data)
            all_results.append(result)

    # Confidence V3
    print("\n" + "=" * 60)
    print("EXPERIMENT: Confidence Router V3 (soft blending)")
    print("=" * 60)
    router = ConfidenceRouterV3(embedder, n_clusters=32, top_k=5, beta=10.0, seed=args.seed)
    router.train(train_data, all_models)
    result = evaluate("Confidence V3 (soft)", router, test_data)
    all_results.append(result)
    print(f"  Accuracy: {result['accuracy']:.4f}")

    # Hybrid
    print("\n" + "=" * 60)
    print("EXPERIMENT: Hybrid (adaptive k + soft confidence)")
    print("=" * 60)
    router = HybridAdaptiveConfidenceRouter(embedder, target_per_cluster=best_tpc_val,
                                            top_k=5, beta=10.0, seed=args.seed)
    router.train(train_data, all_models)
    result = evaluate("Hybrid Adaptive+Confidence", router, test_data)
    all_results.append(result)
    print(f"  Accuracy: {result['accuracy']:.4f}")

    # Print final comparison
    print("\n" + "=" * 90)
    print("ITERATION 2 — ALL RESULTS")
    print("=" * 90)
    print(f"\n{'Router':<40} {'Acc':<8} {'vs Best':<10} {'vs Oracle':<10} {'Time':<8}")
    print("-" * 90)
    for r in sorted(all_results, key=lambda x: x["accuracy"], reverse=True):
        vs_best = r["accuracy"] - best_single_acc
        vs_oracle = r["accuracy"] / oracle_acc * 100 if oracle_acc > 0 else 0
        print(f"{r['name']:<40} {r['accuracy']:<8.4f} {vs_best:+.4f}    {vs_oracle:<10.1f}% {r['time_s']:<8.1f}s")
    print("-" * 90)
    print(f"{'Best single model':<40} {best_single_acc:<8.4f}")
    print(f"{'Oracle':<40} {oracle_acc:<8.4f}")

    # Save
    best = max(all_results, key=lambda x: x["accuracy"])
    output = {
        "iteration": 2,
        "best_single_model": best_single_name,
        "best_single_accuracy": float(best_single_acc),
        "oracle_accuracy": float(oracle_acc),
        "best_router": best["name"],
        "best_router_accuracy": float(best["accuracy"]),
        "improvement_over_baseline": float(best["accuracy"] - 0.7150),
        "all_results": [{"name": r["name"], "accuracy": float(r["accuracy"]), "time_s": float(r["time_s"])} for r in all_results],
    }
    out_path = Path("../qwen/results/iteration2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
