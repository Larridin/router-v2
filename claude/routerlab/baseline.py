"""Baseline: faithful reimplementation of the production router's core algorithm.

AvengersPro cluster scoring (arXiv 2508.12631) with the exact hyperparameters
from the production bundle's metadata.yaml (v0.75): k=16 clusters, top_p=4,
per-prompt z-scored quality, shrinkage k0=10, per-cluster min-max + alpha blend.

The Go scorer computes, per prompt: sum over top-p clusters of
    alpha·qNorm[c] + (1−alpha)·(1−costNorm)
which equals alpha·Σ qNorm[c] + (1−alpha)·p·(1−costNorm). So precompute()
returns Q = Σ qNorm[c] once, and the alpha sweep is a cheap argmax:
    argmax_m alpha·Q[m] + (1−alpha)·cost_scale·(1−costNorm[m]),  cost_scale=p.
"""

import numpy as np
from sklearn.cluster import KMeans

from . import core

K = 16
TOP_P = 4
SHRINKAGE_K0 = 10.0
SEED = 42


class ClusterRouter:
    name = "cluster (AvengersPro reimpl)"

    def __init__(self, k: int = K, top_p: int = TOP_P, k0: float = SHRINKAGE_K0, n_init: int = 10):
        self.k, self.top_p, self.k0, self.n_init = k, top_p, k0, n_init
        self.cost_scale = float(top_p)

    def fit(self, train_emb: np.ndarray, train_quality: np.ndarray, train_cost: np.ndarray):
        z = core.zscore_per_prompt(train_quality)
        km = KMeans(n_clusters=self.k, n_init=self.n_init, random_state=SEED).fit(train_emb)
        centroids = km.cluster_centers_
        self.centroids = centroids / np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12, None)

        labels = km.labels_
        n_models = train_quality.shape[1]
        means = np.zeros((self.k, n_models))
        counts = np.zeros(self.k)
        for c in range(self.k):
            mask = labels == c
            counts[c] = mask.sum()
            if counts[c] > 0:
                means[c] = z[mask].mean(axis=0)
        self.quality_means = core.shrink(means, counts, z.mean(axis=0), self.k0)
        # Per-cluster min-max across models, done once (alpha-independent).
        self.quality_norm = core.minmax_rows(self.quality_means)
        self.cost_norm = core.cost_axis(train_cost.mean(axis=0))
        return self

    def precompute(self, emb: np.ndarray) -> np.ndarray:
        """emb: (n, d) L2-normalized. Returns (n, M) summed top-p normalized quality."""
        sims = emb @ self.centroids.T  # (n, k)
        p = min(self.top_p, self.k)
        top = np.argpartition(-sims, p - 1, axis=1)[:, :p]  # order-free sum
        return self.quality_norm[top].sum(axis=1)
