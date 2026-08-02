"""
Confidence-Aware Cluster Router

Extends the baseline cluster router with:
1. Per-query confidence scoring (cosine similarity to nearest centroid + neighborhood density)
2. OOD detection — low-confidence queries flagged as out-of-distribution
3. Calibrated fallback — uncertain queries fall back to best-single-model
4. Confidence-weighted routing — blend cluster predictions by confidence
"""

import numpy as np
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


class ConfidenceClusterRouter:
    """Cluster router with confidence scoring and OOD-aware fallback."""

    def __init__(
        self,
        embedder,
        n_clusters: int = 32,
        top_k: int = 5,
        beta: float = 10.0,
        confidence_threshold: float = 0.5,
        fallback_mode: str = "best_single",
        seed: int = 42,
    ):
        self.embedder = embedder
        self.n_clusters = n_clusters
        self.top_k = top_k
        self.beta = beta
        self.confidence_threshold = confidence_threshold
        self.fallback_mode = fallback_mode
        self.seed = seed

    def embed_batch(self, texts):
        embs = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            embs.extend(self.embedder.batch(batch, max_batch_size=100))
        return np.array(embs)

    def train(self, train_data, available_models):
        queries = [item["query"] for item in train_data]
        print(f"  [ConfidenceRouter] Embedding {len(queries)} training queries...")
        embs = self.embed_batch(queries)
        print(f"  [ConfidenceRouter] Embeddings: {embs.shape}")

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
        labels = kmeans.fit_predict(embs)

        self.cluster_rankings = {}
        self.cluster_train_embs = {}
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
            self.cluster_train_embs[cid] = embs[mask]

        self.centroids = kmeans.cluster_centers_
        self.available_models = available_models

        # Compute best-single-model from training data for fallback
        model_scores_global = defaultdict(list)
        for item in train_data:
            for m in available_models:
                if m in item["records"] and item["records"][m] is not None:
                    model_scores_global[m].append(float(item["records"][m]))
        self.best_single_model = max(
            model_scores_global,
            key=lambda m: np.mean(model_scores_global[m]) if model_scores_global[m] else 0,
        )

        # Calibrate confidence threshold on training data
        self._calibrate_confidence(embs, labels)

        # Build per-cluster NN for density estimation
        self.cluster_nn = {}
        for cid, c_embs in self.cluster_train_embs.items():
            if len(c_embs) >= 5:
                nn = NearestNeighbors(n_neighbors=min(5, len(c_embs)), metric="cosine", n_jobs=-1)
                nn.fit(c_embs)
                self.cluster_nn[cid] = nn

        print(f"  [ConfidenceRouter] Trained {len(self.cluster_rankings)} clusters")
        print(f"  [ConfidenceRouter] Best single model: {self.best_single_model}")
        print(f"  [ConfidenceRouter] Confidence threshold: {self.confidence_threshold:.4f}")

    def _calibrate_confidence(self, train_embs, train_labels):
        """Calibrate confidence threshold using training set distribution."""
        centroids_norm = self.centroids / (
            np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-12
        )
        embs_norm = train_embs / (np.linalg.norm(train_embs, axis=1, keepdims=True) + 1e-12)
        sims = embs_norm @ centroids_norm.T

        max_sims = np.max(sims, axis=1)
        p25 = np.percentile(max_sims, 25)
        self.confidence_threshold = p25
        print(f"  [ConfidenceRouter] Calibrated threshold to 25th percentile: {p25:.4f}")

    def compute_confidence(self, emb):
        """
        Compute confidence for a single query embedding.
        
        Combines:
        1. Centroid similarity — how close to nearest cluster
        2. Neighborhood density — how many training points nearby
        """
        emb_norm = emb / (np.linalg.norm(emb) + 1e-12)
        centroids_norm = self.centroids / (
            np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-12
        )

        sims = emb_norm @ centroids_norm.T
        top_idx = np.argsort(sims)[-self.top_k:]
        top_sims = sims[top_idx]

        centroid_confidence = np.max(top_sims)

        density_confidence = 0.0
        nearest_cid = top_idx[np.argmax(top_sims)]
        if nearest_cid in self.cluster_nn:
            nn = self.cluster_nn[nearest_cid]
            distances, _ = nn.kneighbors(emb.reshape(1, -1))
            avg_dist = np.mean(distances)
            density_confidence = 1.0 / (1.0 + avg_dist)

        confidence = 0.6 * centroid_confidence + 0.4 * density_confidence
        return confidence, nearest_cid, top_idx, top_sims

    def route(self, queries, return_confidence=False):
        embs = self.embed_batch(queries)
        results = []
        confidences = []
        fallback_count = 0

        for i in range(len(queries)):
            confidence, nearest_cid, top_idx, top_sims = self.compute_confidence(embs[i])
            confidences.append(confidence)

            if confidence < self.confidence_threshold:
                if self.fallback_mode == "best_single":
                    results.append([self.best_single_model])
                else:
                    results.append([self.best_single_model])
                fallback_count += 1
                continue

            weights = np.exp(self.beta * top_sims)
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

            confidence_weight = min(confidence / 0.8, 1.0)
            for m in scores:
                if m == self.best_single_model:
                    scores[m] += (1.0 - confidence_weight) * 0.3

            best = max(scores, key=scores.get)
            results.append([best])

        print(f"  [ConfidenceRouter] Routed {len(queries)} queries, {fallback_count} fell back to best-single ({fallback_count/len(queries)*100:.1f}%)")

        if return_confidence:
            return results, confidences
        return results


class ConfidenceClusterRouterV2:
    """
    V2: More aggressive OOD detection with per-cluster adaptive thresholds.
    Each cluster has its own threshold based on its internal density.
    """

    def __init__(
        self,
        embedder,
        n_clusters: int = 32,
        top_k: int = 5,
        beta: float = 10.0,
        fallback_mode: str = "best_single",
        seed: int = 42,
    ):
        self.embedder = embedder
        self.n_clusters = n_clusters
        self.top_k = top_k
        self.beta = beta
        self.fallback_mode = fallback_mode
        self.seed = seed

    def embed_batch(self, texts):
        embs = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            embs.extend(self.embedder.batch(batch, max_batch_size=100))
        return np.array(embs)

    def train(self, train_data, available_models):
        queries = [item["query"] for item in train_data]
        print(f"  [ConfidenceRouterV2] Embedding {len(queries)} training queries...")
        embs = self.embed_batch(queries)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
        labels = kmeans.fit_predict(embs)

        self.cluster_rankings = {}
        self.cluster_thresholds = {}
        for cid in range(self.n_clusters):
            mask = labels == cid
            if not mask.any():
                continue
            cluster_items = [train_data[i] for i in np.where(mask)[0]]
            cluster_embs = embs[mask]

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
            self.cluster_rankings[cid] = {
                "scores": dict(sorted_m),
                "ranking": [m for m, _ in sorted_m],
                "total": len(cluster_items),
            }

            centroid = kmeans.cluster_centers_[cid]
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
            embs_norm = cluster_embs / (np.linalg.norm(cluster_embs, axis=1, keepdims=True) + 1e-12)
            sims = embs_norm @ centroid_norm
            self.cluster_thresholds[cid] = np.percentile(sims, 15)

        self.centroids = kmeans.cluster_centers_
        self.available_models = available_models

        model_scores_global = defaultdict(list)
        for item in train_data:
            for m in available_models:
                if m in item["records"] and item["records"][m] is not None:
                    model_scores_global[m].append(float(item["records"][m]))
        self.best_single_model = max(
            model_scores_global,
            key=lambda m: np.mean(model_scores_global[m]) if model_scores_global[m] else 0,
        )

        print(f"  [ConfidenceRouterV2] Trained {len(self.cluster_rankings)} clusters with per-cluster thresholds")

    def route(self, queries):
        embs = self.embed_batch(queries)
        embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        centroids_norm = self.centroids / (
            np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-12
        )
        sims = embs_norm @ centroids_norm.T

        results = []
        fallback_count = 0

        for i in range(len(queries)):
            q_sims = sims[i]
            top_idx = np.argsort(q_sims)[-self.top_k:][::-1]
            top_vals = q_sims[top_idx]

            nearest_cid = top_idx[0]
            threshold = self.cluster_thresholds.get(nearest_cid, 0.3)
            is_ood = top_vals[0] < threshold

            if is_ood:
                results.append([self.best_single_model])
                fallback_count += 1
                continue

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

        print(f"  [ConfidenceRouterV2] {fallback_count}/{len(queries)} queries flagged OOD ({fallback_count/len(queries)*100:.1f}%)")
        return results
