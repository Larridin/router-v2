"""Model-agnostic quality, cost, regret, and uncertainty metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Aggregate measurements for one policy at one quality bias."""

    name: str
    quality_bias: float
    mean_quality: float
    mean_cost: float
    best_single_quality_gain: float
    oracle_quality_gap: float
    mean_normalized_oracle_regret: float
    normalized_model_entropy: float
    model_counts: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """Paired mean difference and percentile confidence interval."""

    estimate: float
    low: float
    high: float
    confidence: float
    replicates: int


def selected_outcomes(
    quality: NDArray[np.floating],
    cost: NDArray[np.floating],
    selections: NDArray[np.integer],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return aligned realized quality and cost for model selections."""

    quality_values, cost_values = _outcome_matrices(quality, cost)
    chosen = np.asarray(selections)
    if chosen.ndim != 1 or chosen.shape[0] != quality_values.shape[0]:
        raise ValueError("selections must have one index per outcome row")
    if not np.issubdtype(chosen.dtype, np.integer):
        raise ValueError("selections must contain integer indices")
    chosen = np.asarray(chosen, dtype=np.int64)
    if np.any(chosen < 0) or np.any(chosen >= quality_values.shape[1]):
        raise ValueError("selection indices are outside model columns")
    rows = np.arange(quality_values.shape[0])
    return quality_values[rows, chosen], cost_values[rows, chosen]


def normalized_utility(
    quality: NDArray[np.floating],
    cost: NDArray[np.floating],
    *,
    quality_bias: float,
) -> NDArray[np.float64]:
    """Blend row-normalized quality and inverse cost into [0, 1] utility."""

    _validate_bias(quality_bias)
    quality_values, cost_values = _outcome_matrices(quality, cost)
    quality_score = _row_minmax(quality_values)
    cost_score = _row_minmax(cost_values)
    return quality_bias * quality_score + (1 - quality_bias) * (1 - cost_score)


def model_entropy(selections: NDArray[np.integer], *, n_models: int) -> float:
    """Return normalized Shannon entropy of selected model shares."""

    if n_models <= 0:
        raise ValueError("n_models must be positive")
    chosen = np.asarray(selections)
    if chosen.ndim != 1 or not np.issubdtype(chosen.dtype, np.integer):
        raise ValueError("selections must be a vector of integer indices")
    if chosen.size == 0:
        raise ValueError("selections must not be empty")
    if np.any(chosen < 0) or np.any(chosen >= n_models):
        raise ValueError("selection indices are outside model columns")
    if n_models == 1:
        return 0.0
    counts = np.bincount(chosen.astype(np.int64), minlength=n_models)
    probabilities = counts[counts > 0] / chosen.size
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(n_models)


def evaluate_selection(
    name: str,
    quality: NDArray[np.floating],
    cost: NDArray[np.floating],
    selections: NDArray[np.integer],
    *,
    best_single_index: int,
    quality_bias: float,
) -> EvaluationMetrics:
    """Summarize one policy without using any model-specific assumptions."""

    if not name:
        raise ValueError("name must not be empty")
    quality_values, cost_values = _outcome_matrices(quality, cost)
    if not 0 <= best_single_index < quality_values.shape[1]:
        raise ValueError("best_single_index is outside model columns")
    chosen = np.asarray(selections, dtype=np.int64)
    selected_quality, selected_cost = selected_outcomes(quality_values, cost_values, chosen)
    utility = normalized_utility(quality_values, cost_values, quality_bias=quality_bias)
    rows = np.arange(quality_values.shape[0])
    regret = np.max(utility, axis=1) - utility[rows, chosen]
    model_counts = np.bincount(chosen, minlength=quality_values.shape[1])
    return EvaluationMetrics(
        name=name,
        quality_bias=float(quality_bias),
        mean_quality=float(np.mean(selected_quality)),
        mean_cost=float(np.mean(selected_cost)),
        best_single_quality_gain=float(
            np.mean(selected_quality) - np.mean(quality_values[:, best_single_index])
        ),
        oracle_quality_gap=float(np.mean(np.max(quality_values, axis=1) - selected_quality)),
        mean_normalized_oracle_regret=float(np.mean(regret)),
        normalized_model_entropy=model_entropy(chosen, n_models=quality_values.shape[1]),
        model_counts=tuple(int(count) for count in model_counts),
    )


def pareto_frontier(
    quality: NDArray[np.floating],
    cost: NDArray[np.floating],
) -> NDArray[np.int64]:
    """Return input indices not dominated by higher quality at lower cost."""

    quality_values, cost_values = _point_vectors(quality, cost)
    dominated = np.zeros(quality_values.size, dtype=np.bool_)
    for candidate in range(quality_values.size):
        no_worse = (quality_values >= quality_values[candidate]) & (
            cost_values <= cost_values[candidate]
        )
        strictly_better = (quality_values > quality_values[candidate]) | (
            cost_values < cost_values[candidate]
        )
        dominated[candidate] = bool(np.any(no_worse & strictly_better))
    return np.flatnonzero(~dominated).astype(np.int64)


def normalized_frontier_area(
    quality: NDArray[np.floating],
    cost: NDArray[np.floating],
) -> float:
    """Return trapezoidal area beneath the nondominated normalized frontier."""

    quality_values, cost_values = _point_vectors(quality, cost)
    indices = pareto_frontier(quality_values, cost_values)
    quality_low = float(np.min(quality_values))
    quality_span = float(np.max(quality_values) - quality_low)
    cost_low = float(np.min(cost_values))
    cost_span = float(np.max(cost_values) - cost_low)
    if quality_span <= 1e-12 or cost_span <= 1e-12:
        return 0.0
    normalized_quality = (quality_values[indices] - quality_low) / quality_span
    normalized_cost = (cost_values[indices] - cost_low) / cost_span
    order = np.argsort(normalized_cost, kind="stable")
    sorted_cost = normalized_cost[order]
    sorted_quality = normalized_quality[order]
    unique_cost: list[float] = []
    best_quality: list[float] = []
    for x_value in np.unique(sorted_cost):
        unique_cost.append(float(x_value))
        best_quality.append(float(np.max(sorted_quality[sorted_cost == x_value])))
    return float(np.trapezoid(best_quality, unique_cost))


def cost_saving_at_quality(
    *,
    quality: NDArray[np.floating],
    cost: NDArray[np.floating],
    reference_quality: float,
    reference_cost: float,
) -> float | None:
    """Return fractional cost saving of the cheapest quality-matching point."""

    quality_values, cost_values = _point_vectors(quality, cost)
    if not math.isfinite(reference_quality):
        raise ValueError("reference_quality must be finite")
    if not math.isfinite(reference_cost) or reference_cost <= 0:
        raise ValueError("reference_cost must be positive and finite")
    eligible = cost_values[quality_values >= reference_quality]
    if eligible.size == 0:
        return None
    return float((reference_cost - np.min(eligible)) / reference_cost)


def paired_bootstrap_difference(
    candidate: NDArray[np.floating],
    baseline: NDArray[np.floating],
    *,
    seed: int = 42,
    replicates: int = 2_000,
    confidence: float = 0.95,
) -> BootstrapInterval:
    """Bootstrap a paired candidate-minus-baseline mean difference."""

    candidate_values = _finite_vector(candidate, "candidate")
    baseline_values = _finite_vector(baseline, "baseline")
    if candidate_values.shape != baseline_values.shape:
        raise ValueError("candidate and baseline must align")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    differences = candidate_values - baseline_values
    random = np.random.default_rng(seed)
    indices = random.integers(0, differences.size, size=(replicates, differences.size))
    means = np.mean(differences[indices], axis=1)
    tail = (1 - confidence) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return BootstrapInterval(
        estimate=float(np.mean(differences)),
        low=float(low),
        high=float(high),
        confidence=float(confidence),
        replicates=replicates,
    )


def _outcome_matrices(
    quality: NDArray[np.floating],
    cost: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    quality_values = np.asarray(quality, dtype=np.float64)
    cost_values = np.asarray(cost, dtype=np.float64)
    if quality_values.ndim != 2 or quality_values.shape[0] == 0 or quality_values.shape[1] == 0:
        raise ValueError("quality and cost must be non-empty matrices")
    if quality_values.shape != cost_values.shape:
        raise ValueError("quality and cost shapes must match")
    if not np.all(np.isfinite(quality_values)) or not np.all(np.isfinite(cost_values)):
        raise ValueError("quality and cost must be finite")
    if np.any(cost_values < 0):
        raise ValueError("cost must be non-negative")
    return quality_values, cost_values


def _point_vectors(
    quality: NDArray[np.floating],
    cost: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    quality_values = _finite_vector(quality, "quality")
    cost_values = _finite_vector(cost, "cost")
    if quality_values.shape != cost_values.shape:
        raise ValueError("quality and cost must align")
    if np.any(cost_values < 0):
        raise ValueError("cost must be non-negative")
    return quality_values, cost_values


def _finite_vector(values: NDArray[np.floating], name: str) -> NDArray[np.float64]:
    numbers = np.asarray(values, dtype=np.float64)
    if numbers.ndim != 1 or numbers.size == 0 or not np.all(np.isfinite(numbers)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return numbers


def _row_minmax(values: NDArray[np.float64]) -> NDArray[np.float64]:
    minimum = np.min(values, axis=1, keepdims=True)
    span = np.max(values, axis=1, keepdims=True) - minimum
    return np.divide(values - minimum, span, out=np.zeros_like(values), where=span > 1e-12)


def _validate_bias(quality_bias: float) -> None:
    if not math.isfinite(quality_bias) or not 0 <= quality_bias <= 1:
        raise ValueError("quality_bias must be finite and in [0, 1]")
