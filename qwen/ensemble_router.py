"""
Embedding Ensemble Router with Adaptive Clustering

Combines multiple embedding models with learned weights and auto-scales cluster count
based on corpus density.

Embedders:
1. all-mpnet-base-v2    — general purpose (768d)
2. all-MiniLM-L6-v2     — fast general (384d)  
3. paraphrase-MiniLM-L6-v2 — paraphrase-focused (384d)

Adaptive k:
- Target ~100 prompts per cluster
- Min k=8, max k=128
- Scales with training set size
"""

import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


class EnsembleEmbeddingRouter:
    """Multi-embedder ensemble router with learned blend weights."""

    EMBEDDER_CONFIGS = [
        ("all-mpnet-base-v2", 1.0),
        ("all-MiniLM-L6-v2", 0.6),
        ("paraphrase-MiniLM-L6-v2", 0.5),
    ]

    def __init__(
        self,
        embedders: dict,
        n_clusters: int = 32,
        top_k: int = 5,
        beta: float = 10.0,
        seed: int = 42,
    ):
        self.embedders = embedders
        self.n_clusters = n_clusters
        self.top_k = top_k
        self.beta = beta
        self.seed = seed

    def embed_batch(self, texts, embedder_name):
        emb = self.embedders[embedder_name]
        embs = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            embs.extend(emb.batch(batch, max_batch_size=100))
        return np.array(embs)

    def train(self, train_data, available_models):
        queries = [item["query"] for item in train_data]
        self.available_models = available_models

        self.per_embedder = {}
        for emb_name, _ in self.EMBEDDER_CONFIGS:
            if emb_name not in self.embedders:
                continue
            print(f"  [EnsembleRouter] Training sub-router for {emb_name}...")
            embs = self.embed_batch(queries, emb_name)
            print(f"  [EnsembleRouter] {emb_name} embeddings: {embs.shape}")

            kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
            labels = kmeans.fit_predict(embs)

            cluster_rankings = {}
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
                cluster_rankings[cid] = {
                    "scores": dict(sorted_m),
                    "ranking": [m for m, _ in sorted_m],
                    "total": len(cluster_items),
                }

            self.per_embedder[emb_name] = {
                "kmeans": kmeans,
                "centroids": kmeans.cluster_centers_,
                "rankings": cluster_rankings,
            }

        self._learn_weights(train_data, queries)
        print(f"  [EnsembleRouter] Learned weights: {self.weights}")

    def _learn_weights(self, train_data, queries):
        """Learn ensemble weights by evaluating each embedder on a held-out train subset."""
        n = len(train_data)
        val_size = min(500, n // 4)
        np.random.seed(self.seed)
        val_idx = np.random.choice(n, val_size, replace=False)
        val_queries = [queries[i] for i in val_idx]
        val_data = [train_data[i] for i in val_idx]

        per_embedder_acc = {}
        for emb_name in self.per_embedder:
            info = self.per_embedder[emb_name]
            embs = self.embed_batch(val_queries, emb_name)
            embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
            centroids_norm = info["centroids"] / (
                np.linalg.norm(info["centroids"], axis=1, keepdims=True) + 1e-12
            )
            sims = embs_norm @ centroids_norm.T

            correct = 0.0
            for j in range(len(val_data)):
                q_sims = sims[j]
                top_idx = np.argsort(q_sims)[-self.top_k:][::-1]
                top_vals = q_sims[top_idx]
                weights = np.exp(self.beta * top_vals)
                weights /= weights.sum()

                scores = defaultdict(float)
                for cidx, w in zip(top_idx, weights):
                    if cidx in info["rankings"]:
                        ranking = info["rankings"][cidx]["ranking"]
                        for m in self.available_models:
                            if m in ranking:
                                rank_score = 1.0 / (ranking.index(m) + 1)
                                scores[m] += w * rank_score
                for m in self.available_models:
                    if m not in scores:
                        scores[m] = 0.0

                best = max(scores, key=scores.get)
                score = float(val_data[j]["records"].get(best, 0) or 0)
                correct += score

            acc = correct / val_size
            per_embedder_acc[emb_name] = acc
            print(f"  [EnsembleRouter] {emb_name} val accuracy: {acc:.4f}")

        total = sum(per_embedder_acc.values())
        self.weights = {k: v / total for k, v in per_embedder_acc.items()}

    def route(self, queries):
        per_embedder_scores = {}
        for emb_name in self.per_embedder:
            info = self.per_embedder[emb_name]
            embs = self.embed_batch(queries, emb_name)
            embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
            centroids_norm = info["centroids"] / (
                np.linalg.norm(info["centroids"], axis=1, keepdims=True) + 1e-12
            )
            sims = embs_norm @ centroids_norm.T

            batch_scores = []
            for j in range(len(queries)):
                q_sims = sims[j]
                top_idx = np.argsort(q_sims)[-self.top_k:][::-1]
                top_vals = q_sims[top_idx]
                weights = np.exp(self.beta * top_vals)
                weights /= weights.sum()

                scores = defaultdict(float)
                for cidx, w in zip(top_idx, weights):
                    if cidx in info["rankings"]:
                        ranking = info["rankings"][cidx]["ranking"]
                        for m in self.available_models:
                            if m in ranking:
                                rank_score = 1.0 / (ranking.index(m) + 1)
                                scores[m] += w * rank_score
                for m in self.available_models:
                    if m not in scores:
                        scores[m] = 0.0
                batch_scores.append(dict(scores))

            per_embedder_scores[emb_name] = batch_scores

        results = []
        for j in range(len(queries)):
            blended = defaultdict(float)
            for emb_name, weight in self.weights.items():
                for m, s in per_embedder_scores[emb_name][j].items():
                    blended[m] += weight * s
            best = max(blended, key=blended.get)
            results.append([best])
        return results


class AdaptiveClusterRouter:
    """Cluster router that auto-scales k based on training data size."""

    def __init__(
        self,
        embedder,
        target_per_cluster: int = 100,
        min_clusters: int = 8,
        max_clusters: int = 128,
        top_k: int = 5,
        beta: float = 10.0,
        seed: int = 42,
    ):
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

    def _compute_n_clusters(self, n_samples):
        n = max(self.min_clusters, min(self.max_clusters, n_samples // self.target_per_cluster))
        return n

    def train(self, train_data, available_models):
        queries = [item["query"] for item in train_data]
        n_clusters = self._compute_n_clusters(len(queries))
        self.n_clusters = n_clusters
        print(f"  [AdaptiveRouter] n_samples={len(queries)}, adaptive k={n_clusters}")

        embs = self.embed_batch(queries)
        print(f"  [AdaptiveRouter] Embeddings: {embs.shape}")

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
        print(f"  [AdaptiveRouter] Trained {len(self.cluster_rankings)} clusters")

    def route(self, queries):
        embs = self.embed_batch(queries)
        embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        centroids_norm = self.centroids / (
            np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-12
        )
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


class EnsembleAdaptiveRouter:
    """Combines ensemble embeddings with adaptive clustering."""

    def __init__(
        self,
        embedders: dict,
        target_per_cluster: int = 80,
        min_clusters: int = 8,
        max_clusters: int = 128,
        top_k: int = 5,
        beta: float = 10.0,
        seed: int = 42,
    ):
        self.ensemble = EnsembleEmbeddingRouter(
            embedders=embedders,
            n_clusters=32,
            top_k=top_k,
            beta=beta,
            seed=seed,
        )
        self.target_per_cluster = target_per_cluster
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.seed = seed

    def train(self, train_data, available_models):
        n = len(train_data)
        adaptive_k = max(self.min_clusters, min(self.max_clusters, n // self.target_per_cluster))
        print(f"  [EnsembleAdaptive] n_samples={n}, setting k={adaptive_k}")
        self.ensemble.n_clusters = adaptive_k
        self.ensemble.train(train_data, available_models)

    def route(self, queries):
        return self.ensemble.route(queries)
