from __future__ import annotations

import numpy as np
import pytest

from routerlab.metrics import (
    cost_saving_at_quality,
    evaluate_selection,
    model_entropy,
    normalized_frontier_area,
    normalized_utility,
    paired_bootstrap_difference,
    pareto_frontier,
    selected_outcomes,
)

QUALITY = np.array(
    [
        [0.2, 0.8, 0.5],
        [0.9, 0.3, 0.6],
        [0.4, 0.7, 0.5],
    ],
    dtype=np.float32,
)
COST = np.array(
    [
        [0.1, 0.5, 0.3],
        [0.4, 0.1, 0.2],
        [0.1, 0.4, 0.2],
    ],
    dtype=np.float64,
)


def test_selected_outcomes_and_summary_are_hand_calculable() -> None:
    selected = np.array([1, 0, 2], dtype=np.int64)

    quality, cost = selected_outcomes(QUALITY, COST, selected)
    summary = evaluate_selection(
        "candidate",
        QUALITY,
        COST,
        selected,
        best_single_index=2,
        quality_bias=1.0,
    )

    np.testing.assert_allclose(quality, [0.8, 0.9, 0.5])
    np.testing.assert_allclose(cost, [0.5, 0.4, 0.2])
    assert summary.name == "candidate"
    assert summary.mean_quality == pytest.approx(2.2 / 3)
    assert summary.mean_cost == pytest.approx(1.1 / 3)
    assert summary.best_single_quality_gain == pytest.approx((2.2 - 1.6) / 3)
    assert summary.oracle_quality_gap == pytest.approx((2.4 - 2.2) / 3)
    assert summary.mean_normalized_oracle_regret == pytest.approx(2 / 9)
    assert summary.normalized_model_entropy == pytest.approx(1.0)
    assert summary.model_counts == (1, 1, 1)


def test_normalized_utility_has_exact_quality_and_cost_endpoints() -> None:
    quality_endpoint = normalized_utility(QUALITY, COST, quality_bias=1.0)
    cost_endpoint = normalized_utility(QUALITY, COST, quality_bias=0.0)

    np.testing.assert_allclose(quality_endpoint[0], [0.0, 1.0, 0.5])
    np.testing.assert_allclose(cost_endpoint[0], [1.0, 0.0, 0.5])


def test_entropy_is_normalized_and_handles_unused_models() -> None:
    assert model_entropy(np.array([0, 1, 2]), n_models=3) == pytest.approx(1.0)
    assert model_entropy(np.array([1, 1, 1]), n_models=3) == pytest.approx(0.0)
    assert model_entropy(np.array([0, 0]), n_models=1) == pytest.approx(0.0)


def test_pareto_frontier_uses_both_quality_and_cost() -> None:
    quality = np.array([0.7, 0.8, 0.8, 0.9, 0.6])
    cost = np.array([0.2, 0.3, 0.4, 0.8, 0.5])

    frontier = pareto_frontier(quality, cost)

    assert frontier.tolist() == [0, 1, 3]


def test_normalized_frontier_area_and_cost_saving() -> None:
    quality = np.array([0.0, 0.5, 1.0, 0.4])
    cost = np.array([0.0, 0.5, 1.0, 0.8])

    assert normalized_frontier_area(quality, cost) == pytest.approx(0.5)
    assert cost_saving_at_quality(
        quality=np.array([0.7, 0.8, 0.9]),
        cost=np.array([0.2, 0.5, 0.9]),
        reference_quality=0.8,
        reference_cost=1.0,
    ) == pytest.approx(0.5)
    assert (
        cost_saving_at_quality(
            quality=np.array([0.7]),
            cost=np.array([0.2]),
            reference_quality=0.8,
            reference_cost=1.0,
        )
        is None
    )


def test_paired_bootstrap_is_deterministic_and_paired() -> None:
    candidate = np.array([2.0, 3.0, 5.0, 7.0])
    baseline = np.array([1.0, 2.0, 4.0, 6.0])

    first = paired_bootstrap_difference(candidate, baseline, seed=42, replicates=2_000)
    second = paired_bootstrap_difference(candidate, baseline, seed=42, replicates=2_000)

    assert first == second
    assert first.estimate == pytest.approx(1.0)
    assert first.low == pytest.approx(1.0)
    assert first.high == pytest.approx(1.0)


@pytest.mark.parametrize("selected", [np.array([-1, 0, 1]), np.array([0, 1, 3])])
def test_selection_rejects_out_of_range_indices(selected: np.ndarray) -> None:
    with pytest.raises(ValueError, match="selection indices"):
        selected_outcomes(QUALITY, COST, selected)


def test_metrics_reject_non_finite_input() -> None:
    bad = QUALITY.copy()
    bad[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        selected_outcomes(bad, COST, np.array([0, 0, 0]))
