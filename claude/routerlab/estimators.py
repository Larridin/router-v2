"""Per-prompt, per-model quality estimators on a common calibrated scale.

Each estimator: fit(emb, quality, texts) on the fit fold, then
estimate(emb, texts) -> (n, M) raw scores. calibrate() learns per-model
isotonic maps raw-score -> E[quality] on the calibration fold, so every
estimator speaks "expected quality in [0,1]" — the scale the dollar-EV rule
and the ensemble need. GBT already predicts on that scale; isotonic is a
near-no-op for it and puts everyone through the same pipeline.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from . import core
from .features import text_features

SEED = 42


class _Calibrated:
    def calibrate(self, cal_emb, cal_quality, cal_texts):
        raw = self.estimate(cal_emb, cal_texts)
        self._iso = []
        for m in range(cal_quality.shape[1]):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(raw[:, m], cal_quality[:, m])
            self._iso.append(iso)
        return self

    def estimate_calibrated(self, emb, texts) -> np.ndarray:
        raw = self.estimate(emb, texts)
        out = np.empty_like(raw)
        for m, iso in enumerate(self._iso):
            out[:, m] = iso.predict(raw[:, m])
        return out


class ClusterEstimator(_Calibrated):
    """The champion's estimator at its best setting (k=256), raw-z output."""

    name = "cluster256"

    def __init__(self, k: int = 256, top_p: int = 4, k0: float = 10.0):
        self.k, self.top_p, self.k0 = k, top_p, k0

    def fit(self, emb, quality, texts=None):
        z = core.zscore_per_prompt(quality)
        km = KMeans(n_clusters=self.k, n_init=3, random_state=SEED).fit(emb)
        c = km.cluster_centers_
        self.centroids = c / np.clip(np.linalg.norm(c, axis=1, keepdims=True), 1e-12, None)
        means = np.zeros((self.k, quality.shape[1]))
        counts = np.zeros(self.k)
        for ci in range(self.k):
            mask = km.labels_ == ci
            counts[ci] = mask.sum()
            if counts[ci]:
                means[ci] = z[mask].mean(axis=0)
        self.quality_means = core.shrink(means, counts, z.mean(axis=0), self.k0)
        return self

    def estimate(self, emb, texts=None):
        sims = emb @ self.centroids.T
        p = min(self.top_p, self.k)
        top = np.argpartition(-sims, p - 1, axis=1)[:, :p]
        return self.quality_means[top].mean(axis=1)  # mean (not sum): stable scale


class KNNEstimator(_Calibrated):
    name = "knn64"

    def __init__(self, k: int = 64, sharpness: float = 2.0, k0: float = 10.0):
        self.k, self.sharpness, self.k0 = k, sharpness, k0

    def fit(self, emb, quality, texts=None):
        self.train_emb = emb.astype(np.float32)
        self.z = core.zscore_per_prompt(quality)
        self.global_mean = self.z.mean(axis=0)
        return self

    def estimate(self, emb, texts=None, batch: int = 1024):
        out = np.empty((len(emb), self.z.shape[1]))
        for i in range(0, len(emb), batch):
            b = emb[i : i + batch].astype(np.float32)
            sims = b @ self.train_emb.T
            k = min(self.k, sims.shape[1])
            top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            w = np.clip(sims[np.arange(len(b))[:, None], top], 0.0, None) ** self.sharpness
            wsum = w.sum(axis=1, keepdims=True)
            local = (w[:, :, None] * self.z[top]).sum(axis=1) / np.clip(wsum, 1e-12, None)
            n_eff = np.zeros(len(b))
            np.divide(wsum[:, 0] ** 2, (w**2).sum(axis=1), out=n_eff, where=wsum[:, 0] > 0)
            out[i : i + batch] = core.shrink(local, n_eff, self.global_mean, self.k0)
        return out


class GBTEstimator(_Calibrated):
    """H1: per-model gradient-boosted E[quality | embedding + text features]."""

    name = "gbt"

    def fit(self, emb, quality, texts):
        X = np.hstack([emb, text_features(texts)])
        self.models = []
        for m in range(quality.shape[1]):
            g = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.08, max_leaf_nodes=63,
                early_stopping=True, validation_fraction=0.1, random_state=SEED,
            )
            g.fit(X, quality[:, m])
            self.models.append(g)
        return self

    def estimate(self, emb, texts):
        X = np.hstack([emb, text_features(texts)])
        out = np.column_stack([g.predict(X) for g in self.models])
        return np.clip(out, 0.0, 1.0)


class EnsembleEstimator:
    """H3: average of calibrated member estimates (already on E[quality] scale)."""

    name = "ensemble"

    def __init__(self, members):
        self.members = members

    def estimate_calibrated(self, emb, texts):
        return np.mean([m.estimate_calibrated(emb, texts) for m in self.members], axis=0)
