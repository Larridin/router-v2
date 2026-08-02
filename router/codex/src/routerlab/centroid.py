"""Train and score a transparent semantic centroid router."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import MiniBatchKMeans


@dataclass(frozen=True, slots=True, order=True)
class CentroidConfig:
    """All parameters that affect centroid fitting or inference."""

    n_clusters: int
    top_p: int
    shrinkage: float
    temperature: float
    seed: int

    def __post_init__(self) -> None:
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if self.top_p <= 0 or self.top_p > self.n_clusters:
            raise ValueError("top_p must be in [1, n_clusters]")
        if not math.isfinite(self.shrinkage) or self.shrinkage <= 0:
            raise ValueError("shrinkage must be positive and finite")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True, slots=True)
class Decision:
    """One model selection with inspectable policy metadata."""

    model: str
    model_index: int
    utility: float
    predicted_quality: float
    expected_cost: float
    cluster_ids: tuple[int, ...]
    cluster_weights: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CentroidPolicy:
    """Fitted centroid geometry and per-cluster model quality."""

    models: tuple[str, ...]
    centroids: NDArray[np.float32]
    cluster_quality: NDArray[np.float32]
    expected_cost: NDArray[np.float64]
    config: CentroidConfig

    def __post_init__(self) -> None:
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("models must be non-empty and unique")
        if self.centroids.ndim != 2:
            raise ValueError("centroids must be a matrix")
        expected_quality = (self.centroids.shape[0], len(self.models))
        if self.cluster_quality.shape != expected_quality:
            raise ValueError(
                f"cluster_quality shape {self.cluster_quality.shape}, expected {expected_quality}"
            )
        if self.expected_cost.shape != (len(self.models),):
            raise ValueError("expected_cost shape does not match models")
        if self.centroids.shape[0] != self.config.n_clusters:
            raise ValueError("config n_clusters does not match centroids")
        for name, values in (
            ("centroids", self.centroids),
            ("cluster_quality", self.cluster_quality),
            ("expected_cost", self.expected_cost),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains non-finite values")
        if np.any(self.expected_cost < 0):
            raise ValueError("expected_cost contains negative values")
        normalized = _normalize_rows(self.centroids)
        object.__setattr__(self, "centroids", normalized)
        object.__setattr__(
            self,
            "cluster_quality",
            np.asarray(self.cluster_quality, dtype=np.float32),
        )
        object.__setattr__(self, "expected_cost", np.asarray(self.expected_cost, dtype=np.float64))

    @property
    def dimension(self) -> int:
        """Return the prompt embedding dimension."""

        return int(self.centroids.shape[1])

    def route_vector(
        self,
        vector: NDArray[np.floating],
        *,
        eligible_models: tuple[str, ...] | None = None,
        quality_bias: float,
    ) -> Decision:
        """Select one eligible model for a single prompt embedding."""

        if not math.isfinite(quality_bias) or not 0 <= quality_bias <= 1:
            raise ValueError("quality_bias must be finite and in [0, 1]")
        prompt = np.asarray(vector, dtype=np.float32)
        if prompt.shape != (self.dimension,):
            raise ValueError(f"embedding shape {prompt.shape}, expected {(self.dimension,)}")
        if not np.all(np.isfinite(prompt)):
            raise ValueError("embedding contains non-finite values")
        norm = float(np.linalg.norm(prompt))
        if norm <= 0:
            raise ValueError("embedding must have non-zero norm")
        prompt = prompt / norm

        similarities = self.centroids @ prompt
        cluster_order = np.lexsort((np.arange(len(similarities)), -similarities))
        cluster_ids = cluster_order[: self.config.top_p]
        logits = similarities[cluster_ids].astype(np.float64) / self.config.temperature
        logits -= np.max(logits)
        weights = np.exp(logits)
        weights /= np.sum(weights)
        predicted = weights @ self.cluster_quality[cluster_ids].astype(np.float64)

        eligible_indices = self._eligible_indices(eligible_models)
        eligible_quality = predicted[eligible_indices]
        eligible_cost = self.expected_cost[eligible_indices]
        quality_score = _minmax(eligible_quality)
        cost_score = _minmax(eligible_cost)
        utilities = quality_bias * quality_score + (1 - quality_bias) * (1 - cost_score)
        local_winner = int(np.argmax(utilities))
        winner = eligible_indices[local_winner]
        return Decision(
            model=self.models[winner],
            model_index=winner,
            utility=float(utilities[local_winner]),
            predicted_quality=float(predicted[winner]),
            expected_cost=float(self.expected_cost[winner]),
            cluster_ids=tuple(int(index) for index in cluster_ids),
            cluster_weights=tuple(float(weight) for weight in weights),
        )

    def select_many(
        self,
        embeddings: NDArray[np.floating],
        quality_bias: float,
    ) -> NDArray[np.int64]:
        """Return selected model indices for aligned prompt embeddings."""

        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("embedding matrix dimension does not match policy")
        return np.asarray(
            [self.route_vector(row, quality_bias=quality_bias).model_index for row in values],
            dtype=np.int64,
        )

    def _eligible_indices(self, eligible_models: tuple[str, ...] | None) -> NDArray[np.int64]:
        if eligible_models is None:
            return np.arange(len(self.models), dtype=np.int64)
        if not eligible_models:
            raise ValueError("eligible_models must not be empty")
        if len(set(eligible_models)) != len(eligible_models):
            raise ValueError("eligible_models contains duplicates")
        allowed = set(eligible_models)
        unknown = allowed.difference(self.models)
        if unknown:
            raise ValueError(f"unknown eligible models: {sorted(unknown)}")
        return np.asarray(
            [index for index, model in enumerate(self.models) if model in allowed],
            dtype=np.int64,
        )


@dataclass(frozen=True, slots=True)
class CentroidGeometry:
    """K-means result reusable across inference-only hyperparameters."""

    models: tuple[str, ...]
    centroids: NDArray[np.float32]
    labels: NDArray[np.int64]
    train_quality: NDArray[np.float32]
    expected_cost: NDArray[np.float64]
    seed: int


def shrink_cluster_quality(
    labels: NDArray[np.integer],
    quality: NDArray[np.floating],
    *,
    n_clusters: int,
    shrinkage: float,
) -> NDArray[np.float32]:
    """Estimate each cluster/model mean with a global model prior."""

    assignments = np.asarray(labels, dtype=np.int64)
    outcomes = np.asarray(quality, dtype=np.float64)
    if outcomes.ndim != 2 or assignments.shape != (outcomes.shape[0],):
        raise ValueError("labels and quality rows must align")
    if n_clusters <= 0 or np.any(assignments < 0) or np.any(assignments >= n_clusters):
        raise ValueError("cluster labels are outside the declared range")
    if not math.isfinite(shrinkage) or shrinkage <= 0:
        raise ValueError("shrinkage must be positive and finite")
    if not np.all(np.isfinite(outcomes)):
        raise ValueError("quality contains non-finite values")

    global_mean = np.mean(outcomes, axis=0)
    means = np.empty((n_clusters, outcomes.shape[1]), dtype=np.float64)
    for cluster in range(n_clusters):
        members = outcomes[assignments == cluster]
        means[cluster] = (np.sum(members, axis=0) + shrinkage * global_mean) / (
            len(members) + shrinkage
        )
    return np.asarray(means, dtype=np.float32)


def fit_centroid_policy(
    embeddings: NDArray[np.floating],
    quality: NDArray[np.floating],
    realized_cost: NDArray[np.floating],
    models: tuple[str, ...],
    config: CentroidConfig,
) -> CentroidPolicy:
    """Fit centroid geometry and train-only model statistics."""

    geometry = fit_centroid_geometry(
        embeddings,
        quality,
        realized_cost,
        models,
        n_clusters=config.n_clusters,
        seed=config.seed,
    )
    return build_centroid_policy(
        geometry,
        top_p=config.top_p,
        shrinkage=config.shrinkage,
        temperature=config.temperature,
    )


def fit_centroid_geometry(
    embeddings: NDArray[np.floating],
    quality: NDArray[np.floating],
    realized_cost: NDArray[np.floating],
    models: tuple[str, ...],
    *,
    n_clusters: int,
    seed: int,
) -> CentroidGeometry:
    """Fit K-means once for a family of centroid policy configurations."""

    vectors = _normalize_rows(np.asarray(embeddings, dtype=np.float32))
    outcomes = np.asarray(quality, dtype=np.float32)
    costs = np.asarray(realized_cost, dtype=np.float64)
    expected_shape = (vectors.shape[0], len(models))
    if outcomes.shape != expected_shape or costs.shape != expected_shape:
        raise ValueError("quality and cost must align with embeddings and models")
    if n_clusters <= 0 or vectors.shape[0] < n_clusters:
        raise ValueError("n_clusters cannot exceed training rows")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not np.all(np.isfinite(outcomes)) or not np.all(np.isfinite(costs)):
        raise ValueError("training outcomes contain non-finite values")
    if np.any(costs < 0):
        raise ValueError("training costs contain negative values")

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=min(max(256, n_clusters * 4), vectors.shape[0]),
        n_init=10,
        max_iter=200,
        reassignment_ratio=0.0,
    )
    labels = kmeans.fit_predict(vectors)
    centroids = _normalize_rows(np.asarray(kmeans.cluster_centers_, dtype=np.float32))
    return CentroidGeometry(
        models=models,
        centroids=centroids,
        labels=np.asarray(labels, dtype=np.int64),
        train_quality=outcomes,
        expected_cost=np.mean(costs, axis=0),
        seed=seed,
    )


def build_centroid_policy(
    geometry: CentroidGeometry,
    *,
    top_p: int,
    shrinkage: float,
    temperature: float,
) -> CentroidPolicy:
    """Build one policy from reusable K-means geometry."""

    config = CentroidConfig(
        n_clusters=geometry.centroids.shape[0],
        top_p=top_p,
        shrinkage=shrinkage,
        temperature=temperature,
        seed=geometry.seed,
    )
    cluster_quality = shrink_cluster_quality(
        geometry.labels,
        geometry.train_quality,
        n_clusters=geometry.centroids.shape[0],
        shrinkage=shrinkage,
    )
    return CentroidPolicy(
        models=geometry.models,
        centroids=geometry.centroids,
        cluster_quality=cluster_quality,
        expected_cost=geometry.expected_cost,
        config=config,
    )


def _normalize_rows(values: NDArray[np.floating]) -> NDArray[np.float32]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("vectors must be a finite matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("vectors contain a zero row")
    return np.asarray(matrix / norms, dtype=np.float32)


def _minmax(values: NDArray[np.floating]) -> NDArray[np.float64]:
    numbers = np.asarray(values, dtype=np.float64)
    low = float(np.min(numbers))
    span = float(np.max(numbers) - low)
    if span <= 1e-12:
        return np.zeros_like(numbers)
    return (numbers - low) / span
