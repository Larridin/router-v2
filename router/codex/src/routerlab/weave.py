"""Faithful public-data implementation of Weave's v0.75 cluster recipe."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits


@dataclass(frozen=True, slots=True)
class WeaveConfig:
    """Parameters defining the repository's v0.75 core cluster algorithm."""

    n_clusters: int
    top_p: int
    shrinkage: float
    seed: int
    n_init: int

    def __post_init__(self) -> None:
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if self.top_p <= 0 or self.top_p > self.n_clusters:
            raise ValueError("top_p must be in [1, n_clusters]")
        if not math.isfinite(self.shrinkage) or self.shrinkage <= 0:
            raise ValueError("shrinkage must be positive and finite")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.n_init <= 0:
            raise ValueError("n_init must be positive")

    @classmethod
    def v075(cls) -> WeaveConfig:
        """Return the fixed core recipe declared by the v0.75 bundle."""

        return cls(n_clusters=16, top_p=4, shrinkage=10.0, seed=42, n_init=10)


@dataclass(frozen=True, slots=True)
class WeavePolicy:
    """Fitted Weave cluster quality estimator and quality-cost scorer."""

    models: tuple[str, ...]
    centroids: NDArray[np.float32]
    cluster_quality: NDArray[np.float64]
    expected_cost: NDArray[np.float64]
    config: WeaveConfig
    name: str = field(default="weave_v075", init=False)
    quality_normalized: NDArray[np.float64] = field(init=False, repr=False)
    cost_normalized: NDArray[np.float64] = field(init=False, repr=False)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("models must be non-empty and unique")
        centroids = _normalize_rows(self.centroids)
        quality = _finite_matrix(self.cluster_quality, "cluster_quality")
        cost = np.asarray(self.expected_cost, dtype=np.float64)
        expected_quality_shape = (self.config.n_clusters, len(self.models))
        if centroids.shape[0] != self.config.n_clusters:
            raise ValueError("centroid rows must match n_clusters")
        if quality.shape != expected_quality_shape:
            raise ValueError(
                f"cluster_quality shape {quality.shape}, expected {expected_quality_shape}"
            )
        if cost.shape != (len(self.models),) or not np.all(np.isfinite(cost)):
            raise ValueError("expected_cost must be finite and match models")
        if np.any(cost < 0):
            raise ValueError("expected_cost must be non-negative")
        object.__setattr__(self, "centroids", centroids)
        object.__setattr__(self, "cluster_quality", quality)
        object.__setattr__(self, "expected_cost", cost)
        object.__setattr__(self, "quality_normalized", _row_minmax(quality))
        object.__setattr__(self, "cost_normalized", _minmax(cost))
        object.__setattr__(self, "digest", _policy_digest(self))

    @property
    def dimension(self) -> int:
        """Return the expected embedding dimension."""

        return int(self.centroids.shape[1])

    def select(
        self,
        embeddings: NDArray[np.floating],
        quality_bias: float,
    ) -> NDArray[np.int64]:
        """Select one model per vector using the v0.75 equal-top-P blend."""

        if not math.isfinite(quality_bias) or not 0 <= quality_bias <= 1:
            raise ValueError("quality_bias must be finite and in [0, 1]")
        vectors = _normalize_rows(embeddings)
        if vectors.shape[1] != self.dimension:
            raise ValueError("embedding dimension does not match policy")
        similarities = vectors @ self.centroids.T
        top = np.argsort(-similarities, axis=1, kind="stable")[:, : self.config.top_p]
        quality_score = np.sum(self.quality_normalized[top], axis=1)
        cheapness = self.config.top_p * (1.0 - self.cost_normalized)
        scores = quality_bias * quality_score + (1.0 - quality_bias) * cheapness
        return np.asarray(np.argmax(scores, axis=1), dtype=np.int64)


def zscore_per_prompt(quality: NDArray[np.floating]) -> NDArray[np.float64]:
    """Map clipped per-prompt model z-scores to [0, 1]."""

    values = _finite_matrix(quality, "quality")
    mean = np.mean(values, axis=1, keepdims=True)
    deviation = np.std(values, axis=1, keepdims=True)
    normalized = np.divide(
        values - mean,
        deviation,
        out=np.zeros_like(values),
        where=deviation > 0,
    )
    return (np.clip(normalized, -3.0, 3.0) + 3.0) / 6.0


def fit_weave_policy(
    embeddings: NDArray[np.floating],
    quality: NDArray[np.floating],
    realized_cost: NDArray[np.floating],
    models: tuple[str, ...],
    *,
    config: WeaveConfig | None = None,
) -> WeavePolicy:
    """Fit the fixed Weave recipe using only the supplied training rows."""

    recipe = config or WeaveConfig.v075()
    vectors = _normalize_rows(embeddings)
    outcomes = _finite_matrix(quality, "quality")
    costs = _finite_matrix(realized_cost, "realized_cost")
    expected_shape = (vectors.shape[0], len(models))
    if outcomes.shape != expected_shape or costs.shape != expected_shape:
        raise ValueError("training embeddings, outcomes, and models must align")
    if vectors.shape[0] < recipe.n_clusters:
        raise ValueError("n_clusters cannot exceed training rows")
    if np.any(costs < 0):
        raise ValueError("realized_cost must be non-negative")

    estimator = KMeans(
        n_clusters=recipe.n_clusters,
        n_init=recipe.n_init,
        random_state=recipe.seed,
    )
    with threadpool_limits(limits=1):
        estimator.fit(vectors)
        normalized_centroids = _normalize_rows(estimator.cluster_centers_)
        labels = np.argmax(vectors @ normalized_centroids.T, axis=1)
    normalized_quality = zscore_per_prompt(outcomes)
    global_mean = np.mean(normalized_quality, axis=0)
    cluster_sum = np.zeros((recipe.n_clusters, len(models)), dtype=np.float64)
    cluster_count = np.bincount(labels, minlength=recipe.n_clusters).astype(np.float64)
    np.add.at(cluster_sum, labels, normalized_quality)
    cluster_quality = (cluster_sum + recipe.shrinkage * global_mean[None, :]) / (
        cluster_count[:, None] + recipe.shrinkage
    )
    return WeavePolicy(
        models=models,
        centroids=np.asarray(estimator.cluster_centers_, dtype=np.float32),
        cluster_quality=cluster_quality,
        expected_cost=np.mean(costs, axis=0),
        config=recipe,
    )


def _policy_digest(policy: WeavePolicy) -> str:
    digest = hashlib.sha256(b"routerlab-weave-v075-policy-v1\0")
    metadata = {
        "models": list(policy.models),
        "config": asdict(policy.config),
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for values, dtype in (
        (policy.centroids, "<f4"),
        (policy.cluster_quality, "<f8"),
        (policy.expected_cost, "<f8"),
    ):
        array = np.ascontiguousarray(values, dtype=dtype)
        digest.update(len(array.shape).to_bytes(4, "big"))
        for dimension in array.shape:
            digest.update(dimension.to_bytes(8, "big"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finite_matrix(values: NDArray[np.floating], name: str) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite matrix")
    return matrix


def _normalize_rows(values: NDArray[np.floating]) -> NDArray[np.float32]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("embeddings must be a finite matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("embeddings contain a zero row")
    return np.asarray(matrix / norms, dtype=np.float32)


def _row_minmax(values: NDArray[np.float64]) -> NDArray[np.float64]:
    minimum = np.min(values, axis=1, keepdims=True)
    span = np.max(values, axis=1, keepdims=True) - minimum
    return np.divide(
        values - minimum,
        span,
        out=np.zeros_like(values),
        where=span > 0,
    )


def _minmax(values: NDArray[np.float64]) -> NDArray[np.float64]:
    minimum = float(np.min(values))
    span = float(np.max(values) - minimum)
    if span <= 0:
        return np.zeros_like(values)
    return (values - minimum) / span
