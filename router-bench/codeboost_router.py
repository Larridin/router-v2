"""
Advanced router: uses code-specific embeddings (jina-v2-base-code, same as Weave)
plus XGBoost to predict per-model scores from embeddings.
"""
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
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


class CodeBoostRouter:
    """XGBoost router with code-specific embeddings."""

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
        print(f"  Feature matrix: {X.shape}")

        try:
            import xgboost as xgb
            self.model_type = "xgboost"
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model_type = "sklearn-gb"

        self.models = {}
        self.scalers_y = {}

        for model_name in tqdm(available_models, desc="  Training per-model regressors"):
            y = np.array([
                float(item["records"].get(model_name, 0) or 0)
                for item in train_data
            ])

            if np.std(y) < 1e-6:
                self.models[model_name] = float(np.mean(y))
                continue

            if self.model_type == "xgboost":
                m = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=1.0,
                    reg_lambda=1.0,
                    random_state=self.seed,
                    verbosity=0,
                    n_jobs=-1,
                )
            else:
                m = GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=self.seed,
                )

            m.fit(X, y)
            self.models[model_name] = m

        print(f"  Trained {len(self.models)} models ({self.model_type})")

    def route(self, queries):
        X = self.embed_batch(queries)
        results = []
        batch_size = 500
        for i in range(0, len(queries), batch_size):
            batch = X[i : i + batch_size]
            for j in range(len(batch)):
                scores = {}
                for m in self.available_models:
                    mdl = self.models[m]
                    if isinstance(mdl, float):
                        scores[m] = mdl
                    else:
                        scores[m] = float(mdl.predict(batch[j : j + 1])[0])
                best = max(scores, key=scores.get)
                results.append([best])
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/coding_config.yaml")
    parser.add_argument(
        "--embedding-model",
        default="jinaai/jina-embeddings-v2-base-code",
        help="Code-specific embedding model (same as Weave router uses)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    loader = BaselineDataLoader(config_path=args.config)
    all_records = loader.load_all_records()
    all_models = sorted(set(r.model_name for r in all_records))
    train_records, test_records = loader.split_by_dataset_then_prompt(
        all_records, train_ratio=0.8, random_seed=args.seed
    )
    adaptor = AvengersProAdaptor(config_path=args.config, random_seed=args.seed)
    train_data = adaptor._convert_records_to_jsonl_format(train_records, all_models)
    test_data = adaptor._convert_records_to_jsonl_format(test_records, all_models)
    print(f"Train: {len(train_data)} queries, Test: {len(test_data)} queries")
    print(f"Models: {len(all_models)}")
    print(f"Embedding model: {args.embedding_model}")

    embedder = LocalEmbeddingCache(model_name=args.embedding_model, trust_remote_code=True)

    router = CodeBoostRouter(embedder, seed=args.seed)
    router.train(train_data, all_models)

    result = evaluate("CodeBoost Router", router, test_data, test_records)
    baseline, _ = compute_baseline(test_records)
    model_avgs = {m: np.mean(list(d.values())) if d else 0 for m, d in baseline.items()}
    best_single = max(model_avgs.values())
    oracle_scores = [
        max(float(s) for s in item["records"].values()) for item in test_data
    ]
    oracle = np.mean(oracle_scores)

    print(f"\n{'='*50}")
    print(f"RESULTS: CodeBoost Router")
    print(f"{'='*50}")
    print(f"Router:     {result['accuracy']:.4f}")
    print(f"Best Single: {best_single:.4f} ({max(model_avgs, key=model_avgs.get)})")
    print(f"vs Best:    {result['accuracy'] - best_single:+.4f}")
    print(f"Oracle:     {oracle:.4f}")
    print(f"% Oracle:   {result['accuracy'] / oracle * 100:.1f}%")

    print(f"\nPer-dataset:")
    for ds in sorted(result["dataset_perf"].keys()):
        dp = result["dataset_perf"][ds]
        acc = dp["correct"] / dp["total"] if dp["total"] > 0 else 0
        ds_best = max(
            (baseline[m].get(ds, 0) for m in baseline), default=0
        )
        ds_oracle = np.mean([
            max(float(s) for s in item["records"].values())
            for item in test_data if item["dataset"] == ds
        ])
        print(f"  {ds:<20} {acc:.4f}  (best={ds_best:.4f}, oracle={ds_oracle:.4f})")

    print(f"\nModel selection:")
    total = sum(result["model_counts"].values())
    for model, count in sorted(result["model_counts"].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {model:<40} {count:5d} ({count/total*100:5.1f}%)")


if __name__ == "__main__":
    main()