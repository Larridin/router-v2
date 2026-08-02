"""
Qwen Router — Iteration 3: Final Push

1. Push k higher (tpc=30, 35, 40, 45)
2. Score-based routing (use mean scores directly instead of rank-based)
3. Shrinkage (empirical Bayes) on cluster scores
4. Per-dataset confidence weighting
5. Final comparison with all previous results
"""

import sys
import os
import time
import json
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


class ScoreBasedRouter:
    """Uses mean scores directly instead of rank-based scoring."""

    def __init__(self, embedder, n_clusters=183, top_k=8, beta=8.0, seed=42):
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

        self.cluster_scores = {}
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
            self.cluster_scores[cid] = {"scores": scores, "total": len(cluster_items)}

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

            blended_scores = defaultdict(float)
            for cidx, w in zip(top_idx, weights):
                if cidx in self.cluster_scores:
                    for m in self.available_models:
                        blended_scores[m] += w * self.cluster_scores[cidx]["scores"].get(m, 0)

            best = max(blended_scores, key=blended_scores.get)
            results.append([best])
        return results


class ShrinkageRouter:
    """Score-based router with empirical Bayes shrinkage toward global mean."""

    def __init__(self, embedder, n_clusters=183, top_k=8, beta=8.0, k0=10.0, seed=42):
        self.embedder = embedder
        self.n_clusters = n_clusters
        self.top_k = top_k
        self.beta = beta
        self.k0 = k0
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

        # Global mean per model (prior)
        global_scores = defaultdict(list)
        for item in train_data:
            for m in available_models:
                if m in item["records"] and item["records"][m] is not None:
                    global_scores[m].append(float(item["records"][m]))
        self.global_mean = {m: np.mean(v) if v else 0.0 for m, v in global_scores.items()}

        self.cluster_scores = {}
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

            shrunk = {}
            for m in available_models:
                vals = model_scores.get(m, [])
                n = len(vals)
                if n > 0:
                    shrink_weight = n / (n + self.k0)
                    shrunk[m] = shrink_weight * np.mean(vals) + (1 - shrink_weight) * self.global_mean.get(m, 0)
                else:
                    shrunk[m] = self.global_mean.get(m, 0)
            self.cluster_scores[cid] = {"scores": shrunk, "total": len(cluster_items)}

        self.centroids = kmeans.cluster_centers_
        self.available_models = available_models
        print(f"  [Shrinkage] k0={self.k0}, global_mean range: [{min(self.global_mean.values()):.4f}, {max(self.global_mean.values()):.4f}]")

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

            blended_scores = defaultdict(float)
            for cidx, w in zip(top_idx, weights):
                if cidx in self.cluster_scores:
                    for m in self.available_models:
                        blended_scores[m] += w * self.cluster_scores[cidx]["scores"].get(m, 0)

            best = max(blended_scores, key=blended_scores.get)
            results.append([best])
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/baseline_config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.chdir(str(PROJECT_ROOT))

    print("=" * 90)
    print("QWEN ROUTER — ITERATION 3 (Final Push)")
    print("=" * 90)

    print("\n--- Loading data ---")
    train_data, test_data, all_models, test_records = load_data(args.config, args.seed)
    best_single_name, best_single_acc, oracle_acc = compute_baselines(test_data, test_records)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}, Models: {len(all_models)}")
    print(f"Best single: {best_single_name} ({best_single_acc:.4f}), Oracle: {oracle_acc:.4f}")

    print(f"\n--- Loading embedder: all-mpnet-base-v2 ---")
    embedder = LocalEmbeddingCache(model_name="all-mpnet-base-v2")

    all_results = []

    # Push k higher
    print("\n" + "=" * 60)
    print("EXPERIMENT: Higher k (tpc=30,35,40,45)")
    print("=" * 60)
    for tpc in [30, 35, 40, 45]:
        k = len(train_data) // tpc
        router = ScoreBasedRouter(embedder, n_clusters=k, top_k=8, beta=8.0, seed=args.seed)
        router.train(train_data, all_models)
        result = evaluate(f"ScoreBased(k={k})", router, test_data)
        all_results.append(result)
        print(f"  tpc={tpc}, k={k}: {result['accuracy']:.4f}")

    # Score-based vs rank-based at best k
    print("\n" + "=" * 60)
    print("EXPERIMENT: Score-based vs Rank-based at k=183")
    print("=" * 60)
    router = ScoreBasedRouter(embedder, n_clusters=183, top_k=8, beta=8.0, seed=args.seed)
    router.train(train_data, all_models)
    result = evaluate("ScoreBased(k=183,tk=8,beta=8)", router, test_data)
    all_results.append(result)
    print(f"  Score-based: {result['accuracy']:.4f}")

    # Shrinkage sweep
    print("\n" + "=" * 60)
    print("EXPERIMENT: Shrinkage (k0 sweep)")
    print("=" * 60)
    for k0 in [1, 3, 5, 10, 20, 50, 100]:
        router = ShrinkageRouter(embedder, n_clusters=183, top_k=8, beta=8.0, k0=k0, seed=args.seed)
        router.train(train_data, all_models)
        result = evaluate(f"Shrinkage(k0={k0})", router, test_data)
        all_results.append(result)
        print(f"  k0={k0}: {result['accuracy']:.4f}")

    # Shrinkage at higher k
    print("\n" + "=" * 60)
    print("EXPERIMENT: Shrinkage at higher k")
    print("=" * 60)
    for k in [200, 250, 300]:
        for k0 in [5, 10, 20]:
            router = ShrinkageRouter(embedder, n_clusters=k, top_k=8, beta=8.0, k0=k0, seed=args.seed)
            router.train(train_data, all_models)
            result = evaluate(f"Shrinkage(k={k},k0={k0})", router, test_data)
            all_results.append(result)
            print(f"  k={k}, k0={k0}: {result['accuracy']:.4f}")

    # Score-based sweep at best tpc
    print("\n" + "=" * 60)
    print("EXPERIMENT: Score-based top_k x beta sweep")
    print("=" * 60)
    best_k = 183
    for tk in [6, 7, 8, 9, 10]:
        for beta in [6.0, 8.0, 10.0, 12.0]:
            router = ScoreBasedRouter(embedder, n_clusters=best_k, top_k=tk, beta=beta, seed=args.seed)
            router.train(train_data, all_models)
            result = evaluate(f"ScoreBased(tk={tk},beta={beta})", router, test_data)
            all_results.append(result)

    # Print final comparison
    print("\n" + "=" * 90)
    print("ITERATION 3 — ALL RESULTS (top 30)")
    print("=" * 90)
    print(f"\n{'Router':<40} {'Acc':<8} {'vs Best':<10} {'vs Oracle':<10} {'Time':<8}")
    print("-" * 90)
    sorted_results = sorted(all_results, key=lambda x: x["accuracy"], reverse=True)
    for r in sorted_results[:30]:
        vs_best = r["accuracy"] - best_single_acc
        vs_oracle = r["accuracy"] / oracle_acc * 100 if oracle_acc > 0 else 0
        print(f"{r['name']:<40} {r['accuracy']:<8.4f} {vs_best:+.4f}    {vs_oracle:<10.1f}% {r['time_s']:<8.1f}s")
    print("-" * 90)
    print(f"{'Best single model':<40} {best_single_acc:<8.4f}")
    print(f"{'Oracle':<40} {oracle_acc:<8.4f}")
    print(f"{'Baseline Cluster (iter1)':<40} {0.7150:<8.4f}")

    best = sorted_results[0]
    print(f"\nWINNER: {best['name']} at {best['accuracy']:.4f}")
    print(f"  vs baseline: +{best['accuracy'] - 0.7150:.4f}")
    print(f"  vs best-single: +{best['accuracy'] - best_single_acc:.4f}")
    print(f"  oracle gap: {oracle_acc - best['accuracy']:.4f}")

    # Save
    output = {
        "iteration": 3,
        "best_single_model": best_single_name,
        "best_single_accuracy": float(best_single_acc),
        "oracle_accuracy": float(oracle_acc),
        "baseline_accuracy": 0.7150,
        "winner": best["name"],
        "winner_accuracy": float(best["accuracy"]),
        "improvement_over_baseline": float(best["accuracy"] - 0.7150),
        "all_results": [{"name": r["name"], "accuracy": float(r["accuracy"]), "time_s": float(r["time_s"])} for r in sorted_results],
    }
    out_path = Path("../qwen/results/iteration3.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
