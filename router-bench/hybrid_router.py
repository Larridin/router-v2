"""
Hybrid router: combines semantic embeddings with keyword-based domain features
to capture signals that pure embeddings miss (sentiment, code domains, etc.).
"""

import sys
import re
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from sklearn.cluster import KMeans
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
from run_multi import evaluate, compute_baseline


# Domain keyword patterns
CODE_KEYWORDS = [
    "function", "class ", "def ", "import ", "return ", "python", "javascript",
    "typescript", "rust", "golang", "java", "c++", "algorithm", "leetcode",
    "code", "programming", "debug", "compiler", "runtime", "variable",
    "array", "string", "integer", "boolean", "loop", "recursion",
]
MATH_KEYWORDS = [
    "solve", "equation", "integral", "derivative", "theorem", "proof",
    "sum", "product", "sequence", "polynomial", "matrix", "vector",
    "probability", "statistics", "geometry", "calculus", "algebra",
]
SENTIMENT_KEYWORDS = [
    "feel", "emotion", "happy", "sad", "angry", "fear", "love", "hate",
    "sentiment", "mood", "attitude", "opinion", "reaction", "express",
]
LOGIC_KEYWORDS = [
    "logic", "reason", "deduce", "infer", "valid", "invalid", "premise",
    "conclusion", "syllogism", "fallacy", "paradox", "knights", "knaves",
    "if and only if", "truth table",
]
KNOWLEDGE_KEYWORDS = [
    "what is", "who is", "when did", "where is", "define", "explain",
    "history", "biology", "chemistry", "physics", "geography",
    "capital of", "population", "discovered", "invented",
]

ALL_KEYWORD_SETS = {
    "code": CODE_KEYWORDS,
    "math": MATH_KEYWORDS,
    "sentiment": SENTIMENT_KEYWORDS,
    "logic": LOGIC_KEYWORDS,
    "knowledge": KNOWLEDGE_KEYWORDS,
}


def extract_domain_features(text: str) -> np.ndarray:
    """Extract keyword-based domain features from prompt text."""
    text_lower = text.lower()
    feats = []
    for domain, keywords in ALL_KEYWORD_SETS.items():
        hits = sum(1 for kw in keywords if kw.lower() in text_lower)
        feats.append(hits)
        feats.append(hits / max(len(text_lower.split()), 1))  # normalized
    return np.array(feats, dtype=np.float32)


class HybridRouter:
    """Combines semantic embeddings with keyword-based domain features."""

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

    def _build_features(self, texts):
        sem_embs = self.embed_batch(texts)
        domain_feats = np.array([extract_domain_features(t) for t in texts])
        # Concatenate semantic + domain features
        combined = np.concatenate([sem_embs, domain_feats], axis=1)
        return combined

    def train(self, train_data, available_models):
        queries = [item["query"] for item in train_data]
        print(f"  Building hybrid features for {len(queries)} training queries...")
        feats = self._build_features(queries)
        print(f"  Feature dim: {feats.shape[1]} ({feats.shape[1] - 768} domain features)")

        # Normalize for clustering
        self.scaler = StandardScaler()
        feats_scaled = self.scaler.fit_transform(feats)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
        labels = kmeans.fit_predict(feats_scaled)

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
        feats = self._build_features(queries)
        feats_scaled = self.scaler.transform(feats)

        norms_q = np.linalg.norm(feats_scaled, axis=1, keepdims=True) + 1e-12
        norms_c = np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-12
        sims = (feats_scaled / norms_q) @ (self.centroids / norms_c).T

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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", default="all-mpnet-base-v2")
    parser.add_argument("--nc", type=int, default=32)
    parser.add_argument("--tk", type=int, default=5)
    parser.add_argument("--beta", type=float, default=10.0)
    args = parser.parse_args()

    loader = BaselineDataLoader(config_path="config/baseline_config.yaml")
    all_records = loader.load_all_records()
    all_models = sorted(set(r.model_name for r in all_records))
    train_records, test_records = loader.split_by_dataset_then_prompt(
        all_records, train_ratio=0.8, random_seed=42
    )
    adaptor = AvengersProAdaptor(config_path="config/baseline_config.yaml", random_seed=42)
    train_data = adaptor._convert_records_to_jsonl_format(train_records, all_models)
    test_data = adaptor._convert_records_to_jsonl_format(test_records, all_models)
    print(f"Train: {len(train_data)} queries, Test: {len(test_data)} queries")

    embedder = LocalEmbeddingCache(model_name=args.embedding_model)

    router = HybridRouter(embedder, n_clusters=args.nc, top_k=args.tk, beta=args.beta)
    router.train(train_data, all_models)
    result = evaluate("Hybrid Router", router, test_data, test_records)

    baseline, _ = compute_baseline(test_records)
    model_avgs = {m: np.mean(list(d.values())) if d else 0 for m, d in baseline.items()}
    best_single = max(model_avgs.values())
    oracle_scores = []
    for item in test_data:
        scores = [float(s) for s in item["records"].values()]
        oracle_scores.append(max(scores))
    oracle = np.mean(oracle_scores)

    print(f"\nHybrid Router: {result['accuracy']:.4f}")
    print(f"Best Single:   {best_single:.4f}")
    print(f"vs Best:       {result['accuracy'] - best_single:+.4f}")
    print(f"Oracle:        {oracle:.4f}")
    print(f"vs Oracle:     {result['accuracy'] / oracle * 100:.1f}%")


if __name__ == "__main__":
    main()