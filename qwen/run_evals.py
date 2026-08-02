"""
Qwen Router Experiments — End-to-End Evaluation

Runs and compares:
1. Baseline Cluster Router (from router-bench)
2. Confidence-Aware Cluster Router (V1 + V2)
3. Ensemble Embedding Router
4. Adaptive Cluster Router
5. Ensemble + Adaptive Router

Usage:
    cd router-bench && python ../qwen/run_evals.py
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent / "router-bench"
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.data_loader import BaselineDataLoader
from baselines.adaptors.avengerspro_adaptor import AvengersProAdaptor
from local_embedding import LocalEmbeddingCache

from confidence_router import ConfidenceClusterRouter, ConfidenceClusterRouterV2
from ensemble_router import (
    EnsembleEmbeddingRouter,
    AdaptiveClusterRouter,
    EnsembleAdaptiveRouter,
)


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

    baseline = {}
    for model, datasets in model_dataset_scores.items():
        baseline[model] = {}
        for d, scores in datasets.items():
            baseline[model][d] = sum(scores) / len(scores) if scores else 0.0

    model_avgs = {}
    for model, datasets in baseline.items():
        scores = list(datasets.values())
        if scores:
            model_avgs[model] = np.mean(scores)
    best_single_name = max(model_avgs, key=model_avgs.get) if model_avgs else "?"
    best_single_acc = model_avgs[best_single_name] if model_avgs else 0

    oracle_scores = []
    for item in test_data:
        scores = [float(s) for s in item["records"].values()]
        oracle_scores.append(max(scores))
    oracle_acc = np.mean(oracle_scores)

    return baseline, best_single_name, best_single_acc, oracle_acc


def evaluate(name, router, test_data, test_records):
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

    return {
        "name": name,
        "accuracy": accuracy,
        "raw_accuracy": raw_accuracy,
        "dataset_perf": dict(dataset_perf),
        "model_counts": dict(model_counts),
        "time_s": elapsed,
        "n_queries": n,
    }


def print_comparison(all_results, best_single_name, best_single_acc, oracle_acc, test_records):
    print("\n" + "=" * 90)
    print("QWEN ROUTER EXPERIMENTS — FINAL COMPARISON")
    print("=" * 90)

    print(f"\n{'Router':<35} {'Acc':<8} {'vs Best':<10} {'vs Oracle':<10} {'Time':<10}")
    print("-" * 90)

    for r in sorted(all_results, key=lambda x: x["accuracy"], reverse=True):
        vs_best = r["accuracy"] - best_single_acc
        vs_oracle = r["accuracy"] / oracle_acc * 100 if oracle_acc > 0 else 0
        print(
            f"{r['name']:<35} {r['accuracy']:<8.4f} {vs_best:+.4f}    "
            f"{vs_oracle:<10.1f}% {r['time_s']:<10.1f}s"
        )

    print("-" * 90)
    print(f"{'Best single model':<35} {best_single_acc:<8.4f}  —          —           —")
    print(f"{'Oracle (upper bound)':<35} {oracle_acc:<8.4f}")
    print(f"\nBest single model: {best_single_name}")

    # Per-dataset breakdown
    datasets = sorted(set(r.dataset_id for r in test_records))
    print(f"\n{'Dataset':<20}", end="")
    for r in all_results:
        short = r["name"][:10]
        print(f"{short:<11}", end="")
    print(f"{'BestSgl':<10}{'Oracle':<10}")
    print("-" * (20 + 11 * len(all_results) + 20))

    for ds in datasets:
        print(f"{ds:<20}", end="")
        for r in all_results:
            dp = r["dataset_perf"].get(ds, {"correct": 0, "total": 0})
            acc = dp["correct"] / dp["total"] if dp["total"] > 0 else 0
            print(f"{acc:<11.4f}", end="")

        ds_baselines = {}
        model_dataset_scores = defaultdict(lambda: defaultdict(list))
        for rec in test_records:
            model_dataset_scores[rec.model_name][rec.dataset_id].append(rec.score)
        for model, dsets in model_dataset_scores.items():
            if ds in dsets:
                ds_baselines[model] = sum(dsets[ds]) / len(dsets[ds])
        ds_best = max(ds_baselines.values()) if ds_baselines else 0
        print(f"{ds_best:<10.4f}", end="")

        ds_items = []
        for r in all_results:
            if ds in r["dataset_perf"]:
                break
        print(f"{oracle_acc:<10.4f}")
        print()


def save_results(all_results, best_single_name, best_single_acc, oracle_acc, output_path):
    output = {
        "best_single_model": best_single_name,
        "best_single_accuracy": float(best_single_acc),
        "oracle_accuracy": float(oracle_acc),
        "routers": [],
    }
    for r in all_results:
        entry = {
            "name": r["name"],
            "accuracy": float(r["accuracy"]),
            "raw_accuracy": float(r["raw_accuracy"]),
            "vs_best_single": float(r["accuracy"] - best_single_acc),
            "vs_oracle_pct": float(r["accuracy"] / oracle_acc * 100) if oracle_acc > 0 else 0,
            "time_s": float(r["time_s"]),
            "n_queries": r["n_queries"],
            "dataset_perf": {},
        }
        for ds, perf in r["dataset_perf"].items():
            entry["dataset_perf"][ds] = {
                "accuracy": float(perf["correct"] / perf["total"]) if perf["total"] > 0 else 0,
                "correct": float(perf["correct"]),
                "total": perf["total"],
            }
        output["routers"].append(entry)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Qwen Router Experiments")
    parser.add_argument("--config", default="config/baseline_config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--routers",
        default="all",
        help="Comma-separated: baseline,confidence,confidence_v2,ensemble,adaptive,ensemble_adaptive,all",
    )
    parser.add_argument("--output", default="../qwen/results/comparison.json")
    args = parser.parse_args()

    os.chdir(str(PROJECT_ROOT))

    print("=" * 90)
    print("QWEN ROUTER EXPERIMENTS")
    print("=" * 90)
    print(f"Config: {args.config}")
    print(f"Seed: {args.seed}")

    print("\n--- Loading data ---")
    train_data, test_data, all_models, test_records = load_data(args.config, args.seed)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}, Models: {len(all_models)}")

    baseline, best_single_name, best_single_acc, oracle_acc = compute_baselines(test_data, test_records)
    print(f"Best single model: {best_single_name} ({best_single_acc:.4f})")
    print(f"Oracle: {oracle_acc:.4f}")

    routers_to_run = args.routers.split(",")
    all_results = []

    # Load primary embedder
    print(f"\n--- Loading primary embedder: all-mpnet-base-v2 ---")
    primary_embedder = LocalEmbeddingCache(model_name="all-mpnet-base-v2")

    # ====== Baseline Cluster Router ======
    if "all" in routers_to_run or "baseline" in routers_to_run:
        print("\n" + "=" * 60)
        print("BASELINE: Cluster Router (all-mpnet-base-v2, k=32)")
        print("=" * 60)
        from run_multi import TunedClusterRouter
        router = TunedClusterRouter(primary_embedder, n_clusters=32, top_k=5, beta=10.0, seed=args.seed)
        router.train(train_data, all_models)
        result = evaluate("Baseline Cluster", router, test_data, test_records)
        all_results.append(result)
        print(f"  Accuracy: {result['accuracy']:.4f}")

    # ====== Confidence Router V1 ======
    if "all" in routers_to_run or "confidence" in routers_to_run:
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: Confidence-Aware Cluster Router V1")
        print("=" * 60)
        router = ConfidenceClusterRouter(
            primary_embedder, n_clusters=32, top_k=5, beta=10.0, seed=args.seed
        )
        router.train(train_data, all_models)
        result = evaluate("Confidence Router V1", router, test_data, test_records)
        all_results.append(result)
        print(f"  Accuracy: {result['accuracy']:.4f}")

    # ====== Confidence Router V2 ======
    if "all" in routers_to_run or "confidence_v2" in routers_to_run:
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: Confidence-Aware Cluster Router V2 (per-cluster thresholds)")
        print("=" * 60)
        router = ConfidenceClusterRouterV2(
            primary_embedder, n_clusters=32, top_k=5, beta=10.0, seed=args.seed
        )
        router.train(train_data, all_models)
        result = evaluate("Confidence Router V2", router, test_data, test_records)
        all_results.append(result)
        print(f"  Accuracy: {result['accuracy']:.4f}")

    # ====== Ensemble Router ======
    if "all" in routers_to_run or "ensemble" in routers_to_run:
        print("\n" + "=" * 60)
        print("EXPERIMENT 3: Ensemble Embedding Router")
        print("=" * 60)
        embedders = {"all-mpnet-base-v2": primary_embedder}
        for emb_name, _ in EnsembleEmbeddingRouter.EMBEDDER_CONFIGS[1:]:
            print(f"  Loading {emb_name}...")
            embedders[emb_name] = LocalEmbeddingCache(model_name=emb_name)

        router = EnsembleEmbeddingRouter(
            embedders=embedders, n_clusters=32, top_k=5, beta=10.0, seed=args.seed
        )
        router.train(train_data, all_models)
        result = evaluate("Ensemble Router", router, test_data, test_records)
        all_results.append(result)
        print(f"  Accuracy: {result['accuracy']:.4f}")

    # ====== Adaptive Cluster Router ======
    if "all" in routers_to_run or "adaptive" in routers_to_run:
        print("\n" + "=" * 60)
        print("EXPERIMENT 4: Adaptive Cluster Router")
        print("=" * 60)
        router = AdaptiveClusterRouter(
            primary_embedder, target_per_cluster=100, min_clusters=8, max_clusters=128,
            top_k=5, beta=10.0, seed=args.seed
        )
        router.train(train_data, all_models)
        result = evaluate("Adaptive Cluster", router, test_data, test_records)
        all_results.append(result)
        print(f"  Accuracy: {result['accuracy']:.4f}")

    # ====== Ensemble + Adaptive Router ======
    if "all" in routers_to_run or "ensemble_adaptive" in routers_to_run:
        print("\n" + "=" * 60)
        print("EXPERIMENT 5: Ensemble + Adaptive Router")
        print("=" * 60)
        embedders = {"all-mpnet-base-v2": primary_embedder}
        for emb_name, _ in EnsembleEmbeddingRouter.EMBEDDER_CONFIGS[1:]:
            if emb_name not in embedders:
                print(f"  Loading {emb_name}...")
                embedders[emb_name] = LocalEmbeddingCache(model_name=emb_name)

        router = EnsembleAdaptiveRouter(
            embedders=embedders, target_per_cluster=80, min_clusters=8, max_clusters=128,
            top_k=5, beta=10.0, seed=args.seed
        )
        router.train(train_data, all_models)
        result = evaluate("Ensemble+Adaptive", router, test_data, test_records)
        all_results.append(result)
        print(f"  Accuracy: {result['accuracy']:.4f}")

    # ====== Print final comparison ======
    print_comparison(all_results, best_single_name, best_single_acc, oracle_acc, test_records)

    # ====== Save results ======
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(all_results, best_single_name, best_single_acc, oracle_acc, str(output_path))


if __name__ == "__main__":
    main()
