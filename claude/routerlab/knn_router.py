"""New algorithm: kNN-local quality estimation.

Same inputs, same cost blend, same alpha semantics as the baseline — the only
change is the quality estimate. Instead of quantizing quality to 16 cluster
means, estimate each model's quality for THIS prompt from its K nearest
training prompts (similarity-weighted), shrunk toward the model's global mean.
Cluster means are a coarse, precomputed form of exactly this; going local is
the hypothesis under test.

precompute() returns the min-max-normalized local quality matrix; the sweep is
    argmax_m alpha·Q[m] + (1−alpha)·cost_scale·(1−costNorm[m]),  cost_scale=1.
"""

import numpy as np

from . import core

K_NEIGHBORS = 64
SIM_SHARPNESS = 8.0  # weight = max(cos, 0)^sharpness
SHRINKAGE_K0 = 10.0


class KNNRouter:
    name = f"knn (k={K_NEIGHBORS} local quality)"
    cost_scale = 1.0

    def __init__(self, k: int = K_NEIGHBORS, sharpness: float = SIM_SHARPNESS, k0: float = SHRINKAGE_K0):
        self.k, self.sharpness, self.k0 = k, sharpness, k0

    def fit(self, train_emb: np.ndarray, train_quality: np.ndarray, train_cost: np.ndarray):
        self.train_emb = train_emb.astype(np.float32)
        self.z = core.zscore_per_prompt(train_quality)
        self.global_mean = self.z.mean(axis=0)
        self.cost_norm = core.cost_axis(train_cost.mean(axis=0))
        return self

    def precompute(self, emb: np.ndarray, batch: int = 1024) -> np.ndarray:
        """emb: (n, d) L2-normalized. Returns (n, M) normalized local quality."""
        out = np.empty((len(emb), self.z.shape[1]))
        for i in range(0, len(emb), batch):
            b = emb[i : i + batch].astype(np.float32)
            sims = b @ self.train_emb.T  # (b, n_train)
            k = min(self.k, sims.shape[1])
            top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            rows = np.arange(len(b))[:, None]
            w = np.clip(sims[rows, top], 0.0, None) ** self.sharpness  # (b, k)
            wsum = w.sum(axis=1, keepdims=True)
            local = (w[:, :, None] * self.z[top]).sum(axis=1) / np.clip(wsum, 1e-12, None)
            # Shrink by effective sample size (Kish) so a low-similarity
            # neighborhood defers to the global mean instead of trusting noise.
            n_eff = np.zeros(len(b))
            np.divide(wsum[:, 0] ** 2, (w**2).sum(axis=1), out=n_eff, where=wsum[:, 0] > 0)
            out[i : i + batch] = core.shrink(local, n_eff, self.global_mean, self.k0)
        return core.minmax_rows(out)
