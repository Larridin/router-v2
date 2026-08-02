"""
LightGBM LambdaRank router: trains a learning-to-rank model that predicts
the best model ordering per query from its embedding.
"""
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
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


class LGBRankRouter:
    """LambdaRank model that learns to rank models per query from embeddings."""

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

        # Build ranking dataset: one row per (query, model) pair
        X_list = []
        y_list = []
        qid_list = []

        query_embs = self.embed_batch(queries)

        self.model_to_idx = {m: i for i, m in enumerate(available_models)}
        self.idx_to_model = {i: m for m, i in self.model_to_idx.items()}

        for qid, (emb, item) in enumerate(zip(query_embs, train_data)):
            for model in available_models:
                score = float(item["records"].get(model, 0) or 0)
                # Features: embedding + one-hot model indicator
                model_onehot = np.zeros(len(available_models))
                model_onehot[self.model_to_idx[model]] = 1.0
                feat = np.concatenate([emb, model_onehot])
                X_list.append(feat)
                y_list.append(score)
                qid_list.append(qid)

        X = np.array(X_list)
        y = np.array(y_list)
        qids = np.array(qid_list)

        print(f"  Training LightGBM ranker ({len(X)} rows, {X.shape[1]} features)...")

        try:
            import lightgbm as lgb
            self.clf = lgb.LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                boosting_type="gbdt",
                n_estimators=200,
                num_leaves=31,
                learning_rate=0.05,
                min_child_samples=20,
                random_state=self.seed,
                verbosity=-1,
            )
            self.clf.fit(X, y, group=[len(available_models)] * len(train_data))
            self.model_type = "lightgbm"
            print(f"  LightGBM ranker trained successfully")
        except ImportError:
            print("  lightgbm not installed, falling back to direct cluster scores")
            # Fallback: just use the per-model mean scores
            self.fallback_scores = {}
            for model in available_models:
                scores = [float(item["records"].get(model, 0) or 0) for item in train_data]
                self.fallback_scores[model] = np.mean(scores)
            self.model_type = "fallback"

    def route(self, queries):
        embs = self.embed_batch(queries)
        results = []

        if self.model_type == "lightgbm":
            for emb in embs:
                # Score each model for this query
                feats = []
                for model in self.available_models:
                    model_onehot = np.zeros(len(self.available_models))
                    model_onehot[self.model_to_idx[model]] = 1.0
                    feats.append(np.concatenate([emb, model_onehot]))
                scores = self.clf.predict(np.array(feats))
                best_idx = np.argmax(scores)
                results.append([self.idx_to_model[best_idx]])
        else:
            # Fallback: always pick the model with highest mean score
            best = max(self.fallback_scores, key=self.fallback_scores.get)
            results = [[best]] * len(queries)

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/coding_config.yaml")
    parser.add_argument("--embedding-model", default="all-mpnet-base-v2")
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

    baseline, _ = compute_baseline(test_records)
    model_avgs = {m: np.mean(list(d.values())) if d else 0 for m, d in baseline.items()}
    best_single = max(model_avgs.values())
    oracle = np.mean([max(float(s) for s in item["records"].values()) for item in test_data])

    embedder = LocalEmbeddingCache(model_name=args.embedding_model)
    router = LGBRankRouter(embedder, seed=args.seed)
    router.train(train_data, all_models)
    result = evaluate("LGBRank Router", router, test_data, test_records)

    print(f"\nRouter:     {result['accuracy']:.4f}")
    print(f"Best Single: {best_single:.4f}")
    print(f"vs Best:    {result['accuracy'] - best_single:+.4f}")
    print(f"Oracle:     {oracle:.4f}")
    print(f"% Oracle:   {result['accuracy'] / oracle * 100:.1f}%")

    print(f"\nModel selection:")
    total = sum(result["model_counts"].values())
    for m, c in sorted(result["model_counts"].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {m:<40} {c:5d} ({c/total*100:5.1f}%)")


if __name__ == "__main__":
    main()