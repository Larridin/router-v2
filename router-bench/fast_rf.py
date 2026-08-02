"""
Fast ensemble router: trains a single Random Forest classifier per query
to predict the best model, plus tries larger embedding models.
"""
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.data_loader import BaselineDataLoader
from baselines.adaptors.avengerspro_adaptor import AvengersProAdaptor
from local_embedding import LocalEmbeddingCache
from run_multi import evaluate, compute_baseline


def load(args):
    loader = BaselineDataLoader(config_path=args.config)
    all_records = loader.load_all_records()
    all_models = sorted(set(r.model_name for r in all_records))
    train_records, test_records = loader.split_by_dataset_then_prompt(
        all_records, train_ratio=0.8, random_seed=args.seed
    )
    adaptor = AvengersProAdaptor(config_path=args.config, random_seed=args.seed)
    train_data = adaptor._convert_records_to_jsonl_format(train_records, all_models)
    test_data = adaptor._convert_records_to_jsonl_format(test_records, all_models)
    return train_data, test_data, all_models, test_records


class RFCClassifier:
    """Single Random Forest classifier predicting best model."""

    def __init__(self, embedder, seed=42):
        self.embedder = embedder
        self.seed = seed

    def embed_batch(self, texts):
        embs = []
        for i in range(0, len(texts), 100):
            embs.extend(self.embedder.batch(texts[i : i + 100], max_batch_size=100))
        return np.array(embs)

    def train(self, train_data, available_models):
        self.available_models = available_models
        queries = [item["query"] for item in train_data]
        X = self.embed_batch(queries)

        # Label: best model per query
        y = []
        for item in train_data:
            best_m = max(available_models, key=lambda m: float(item["records"].get(m, 0) or 0))
            y.append(best_m)

        self.label_to_idx = {m: i for i, m in enumerate(available_models)}
        self.idx_to_label = {i: m for m, i in self.label_to_idx.items()}
        y_int = np.array([self.label_to_idx[m] for m in y])

        print(f"  Training RF classifier ({len(available_models)} classes)...")
        self.clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=20,
            min_samples_split=50,
            max_features=0.5,
            random_state=self.seed,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.clf.fit(X, y_int)
        print(f"  Train accuracy: {self.clf.score(X, y_int):.4f}")

    def route(self, queries):
        X = self.embed_batch(queries)
        preds = self.clf.predict(X)
        return [[self.idx_to_label[p]] for p in preds]


class GBCClassifier:
    """Gradient Boosting classifier predicting best model."""

    def __init__(self, embedder, seed=42):
        self.embedder = embedder
        self.seed = seed

    def embed_batch(self, texts):
        embs = []
        for i in range(0, len(texts), 100):
            embs.extend(self.embedder.batch(texts[i : i + 100], max_batch_size=100))
        return np.array(embs)

    def train(self, train_data, available_models):
        self.available_models = available_models
        queries = [item["query"] for item in train_data]
        X = self.embed_batch(queries)

        y = []
        for item in train_data:
            best_m = max(available_models, key=lambda m: float(item["records"].get(m, 0) or 0))
            y.append(best_m)

        self.label_to_idx = {m: i for i, m in enumerate(available_models)}
        self.idx_to_label = {i: m for m, i in self.label_to_idx.items()}
        y_int = np.array([self.label_to_idx[m] for m in y])

        print(f"  Training GB classifier ({len(available_models)} classes)...")
        self.clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.seed,
        )
        self.clf.fit(X, y_int)
        print(f"  Train accuracy: {self.clf.score(X, y_int):.4f}")

    def route(self, queries):
        X = self.embed_batch(queries)
        preds = self.clf.predict(X)
        return [[self.idx_to_label[p]] for p in preds]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/coding_config.yaml")
    parser.add_argument("--embedding-model", default="all-mpnet-base-v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", default="rf", choices=["rf", "gb"])
    args = parser.parse_args()

    train_data, test_data, all_models, test_records = load(args)
    print(f"Train: {len(train_data)} queries, Test: {len(test_data)} queries")

    baseline, _ = compute_baseline(test_records)
    model_avgs = {m: np.mean(list(d.values())) if d else 0 for m, d in baseline.items()}
    best_single = max(model_avgs.values())
    oracle_scores = [
        max(float(s) for s in item["records"].values()) for item in test_data
    ]
    oracle = np.mean(oracle_scores)

    embedder = LocalEmbeddingCache(model_name=args.embedding_model)

    if args.method == "rf":
        router = RFCClassifier(embedder, seed=args.seed)
        name = "RandomForest"
    else:
        router = GBCClassifier(embedder, seed=args.seed)
        name = "GradientBoost"

    router.train(train_data, all_models)
    result = evaluate(f"{name} Router", router, test_data, test_records)

    print(f"\n{'='*50}")
    print(f"RESULTS: {name} Router")
    print(f"{'='*50}")
    print(f"Router:     {result['accuracy']:.4f}")
    print(f"Best Single: {best_single:.4f}")
    print(f"vs Best:    {result['accuracy'] - best_single:+.4f}")
    print(f"Oracle:     {oracle:.4f}")
    print(f"% Oracle:   {result['accuracy'] / oracle * 100:.1f}%")

    print(f"\nModel selection:")
    total = sum(result["model_counts"].values())
    for model, count in sorted(result["model_counts"].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {model:<40} {count:5d} ({count/total*100:5.1f}%)")


if __name__ == "__main__":
    main()