"""Comparable train-only routing baselines and an evaluator oracle."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge


class VectorPolicy(Protocol):
    """A routing policy operating on aligned prompt embeddings."""

    name: str

    def select(
        self,
        embeddings: NDArray[np.floating],
        quality_bias: float,
    ) -> NDArray[np.int64]:
        """Return one model-column index per embedding row."""

        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RandomPolicy:
    """Deterministic pseudo-random selection baseline."""

    n_models: int
    seed: int
    name: str = "random"

    def __post_init__(self) -> None:
        if self.n_models <= 0:
            raise ValueError("n_models must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    def select(
        self,
        embeddings: NDArray[np.floating],
        quality_bias: float,
    ) -> NDArray[np.int64]:
        """Hash row content and position to avoid mutable random state."""

        _validate_bias(quality_bias)
        values = _matrix(embeddings)
        selected = np.empty(values.shape[0], dtype=np.int64)
        for index, row in enumerate(values):
            digest = hashlib.sha256()
            digest.update(b"routerlab-random-v1\0")
            digest.update(self.seed.to_bytes(8, "big"))
            digest.update(index.to_bytes(8, "big"))
            digest.update(np.asarray(row, dtype="<f4").tobytes())
            selected[index] = int.from_bytes(digest.digest()[:8], "big") % self.n_models
        return selected


@dataclass(frozen=True, slots=True)
class ConstantPolicy:
    """Always select one train-derived model index."""

    model_index: int
    n_models: int
    name: str

    def select(
        self,
        embeddings: NDArray[np.floating],
        quality_bias: float,
    ) -> NDArray[np.int64]:
        """Return the fixed model for every prompt row."""

        _validate_bias(quality_bias)
        rows = _matrix(embeddings).shape[0]
        return np.full(rows, self.model_index, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class GlobalUtilityPolicy:
    """Use train-global quality and cost without prompt features."""

    global_quality: NDArray[np.float64]
    expected_cost: NDArray[np.float64]
    name: str = "global_utility"

    @classmethod
    def fit(
        cls,
        train_quality: NDArray[np.floating],
        train_cost: NDArray[np.floating],
    ) -> GlobalUtilityPolicy:
        """Fit model-global statistics from training rows."""

        quality, cost = _aligned_outcomes(train_quality, train_cost)
        return cls(np.mean(quality, axis=0), np.mean(cost, axis=0))

    def select(
        self,
        embeddings: NDArray[np.floating],
        quality_bias: float,
    ) -> NDArray[np.int64]:
        """Select one global utility winner for all input rows."""

        rows = _matrix(embeddings).shape[0]
        quality = np.tile(self.global_quality, (rows, 1))
        return select_by_predicted_quality(quality, self.expected_cost, quality_bias)


@dataclass(frozen=True, slots=True)
class RidgePolicy:
    """Multi-output linear quality predictor over prompt embeddings."""

    estimator: Ridge
    expected_cost: NDArray[np.float64]
    name: str = "ridge"

    def select(
        self,
        embeddings: NDArray[np.floating],
        quality_bias: float,
    ) -> NDArray[np.int64]:
        """Predict per-model quality and maximize blended utility."""

        vectors = _normalize_rows(embeddings)
        predicted = np.asarray(self.estimator.predict(vectors), dtype=np.float64)
        return select_by_predicted_quality(predicted, self.expected_cost, quality_bias)


@dataclass(frozen=True, slots=True)
class KNNPolicy:
    """Average full-information outcomes of nearby training prompts."""

    train_embeddings: NDArray[np.float32]
    train_quality: NDArray[np.float64]
    expected_cost: NDArray[np.float64]
    neighbors: int
    name: str = "knn"

    def select(
        self,
        embeddings: NDArray[np.floating],
        quality_bias: float,
    ) -> NDArray[np.int64]:
        """Route from cosine-nearest train rows only."""

        query = _normalize_rows(embeddings)
        similarities = query @ self.train_embeddings.T
        order = np.argsort(-similarities, axis=1, kind="stable")[:, : self.neighbors]
        predicted = np.mean(self.train_quality[order], axis=1)
        return select_by_predicted_quality(predicted, self.expected_cost, quality_bias)


def constant_baselines(
    train_quality: NDArray[np.floating],
    train_cost: NDArray[np.floating],
) -> tuple[ConstantPolicy, ConstantPolicy]:
    """Fit cheapest and best-single constants from training outcomes."""

    quality, cost = _aligned_outcomes(train_quality, train_cost)
    n_models = quality.shape[1]
    cheapest = int(np.argmin(np.mean(cost, axis=0)))
    best = int(np.argmax(np.mean(quality, axis=0)))
    return (
        ConstantPolicy(cheapest, n_models, "cheapest"),
        ConstantPolicy(best, n_models, "best_single"),
    )


def fit_ridge_policy(
    train_embeddings: NDArray[np.floating],
    train_quality: NDArray[np.floating],
    train_cost: NDArray[np.floating],
    *,
    alpha: float = 1.0,
) -> RidgePolicy:
    """Fit a multi-output ridge quality predictor on training rows."""

    vectors = _normalize_rows(train_embeddings)
    quality, cost = _aligned_outcomes(train_quality, train_cost)
    if vectors.shape[0] != quality.shape[0]:
        raise ValueError("training embeddings and outcomes must align")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("ridge alpha must be finite and non-negative")
    estimator = Ridge(alpha=alpha)
    estimator.fit(vectors, quality)
    return RidgePolicy(estimator, np.mean(cost, axis=0))


def fit_knn_policy(
    train_embeddings: NDArray[np.floating],
    train_quality: NDArray[np.floating],
    train_cost: NDArray[np.floating],
    *,
    neighbors: int = 5,
) -> KNNPolicy:
    """Fit an exact cosine kNN policy from training rows."""

    vectors = _normalize_rows(train_embeddings)
    quality, cost = _aligned_outcomes(train_quality, train_cost)
    if vectors.shape[0] != quality.shape[0]:
        raise ValueError("training embeddings and outcomes must align")
    if neighbors <= 0 or neighbors > vectors.shape[0]:
        raise ValueError("neighbors must be in [1, training rows]")
    return KNNPolicy(vectors, quality, np.mean(cost, axis=0), neighbors)


def oracle_select(
    held_out_quality: NDArray[np.floating],
    held_out_cost: NDArray[np.floating],
    *,
    quality_bias: float,
) -> NDArray[np.int64]:
    """Select from held-out outcomes as a non-servable upper bound."""

    quality, cost = _aligned_outcomes(held_out_quality, held_out_cost)
    return select_by_predicted_quality(quality, cost, quality_bias)


def select_by_predicted_quality(
    predicted_quality: NDArray[np.floating],
    expected_cost: NDArray[np.floating],
    quality_bias: float,
) -> NDArray[np.int64]:
    """Blend row-wise normalized predicted quality and expected cost."""

    _validate_bias(quality_bias)
    quality = _matrix(predicted_quality).astype(np.float64)
    cost = np.asarray(expected_cost, dtype=np.float64)
    if cost.ndim == 1:
        if cost.shape != (quality.shape[1],):
            raise ValueError("expected_cost does not match model columns")
        cost = np.tile(cost, (quality.shape[0], 1))
    if cost.shape != quality.shape:
        raise ValueError("expected_cost must be per-model or match predicted quality")
    if not np.all(np.isfinite(cost)) or np.any(cost < 0):
        raise ValueError("expected_cost must be finite and non-negative")
    quality_score = _row_minmax(quality)
    cost_score = _row_minmax(cost)
    utility = quality_bias * quality_score + (1 - quality_bias) * (1 - cost_score)
    return np.asarray(np.argmax(utility, axis=1), dtype=np.int64)


def _aligned_outcomes(
    quality: NDArray[np.floating],
    cost: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    quality_values = _matrix(quality).astype(np.float64)
    cost_values = _matrix(cost).astype(np.float64)
    if quality_values.shape != cost_values.shape:
        raise ValueError("quality and cost shapes must match")
    if np.any(cost_values < 0):
        raise ValueError("cost must be non-negative")
    return quality_values, cost_values


def _matrix(values: NDArray[np.floating]) -> NDArray[np.floating]:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("values must be a finite matrix")
    return matrix


def _normalize_rows(values: NDArray[np.floating]) -> NDArray[np.float32]:
    matrix = np.asarray(_matrix(values), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("embeddings contain a zero row")
    return np.asarray(matrix / norms, dtype=np.float32)


def _row_minmax(values: NDArray[np.float64]) -> NDArray[np.float64]:
    minimum = np.min(values, axis=1, keepdims=True)
    span = np.max(values, axis=1, keepdims=True) - minimum
    return np.divide(values - minimum, span, out=np.zeros_like(values), where=span > 1e-12)


def _validate_bias(quality_bias: float) -> None:
    if not math.isfinite(quality_bias) or not 0 <= quality_bias <= 1:
        raise ValueError("quality_bias must be finite and in [0, 1]")
