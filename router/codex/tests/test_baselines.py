from __future__ import annotations

import numpy as np

from routerlab.baselines import (
    GlobalUtilityPolicy,
    RandomPolicy,
    constant_baselines,
    fit_knn_policy,
    fit_ridge_policy,
    oracle_select,
)


def training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=np.float32,
    )
    quality = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=np.float32,
    )
    cost = np.array(
        [[0.3, 0.1], [0.3, 0.1], [0.3, 0.1], [0.3, 0.1]],
        dtype=np.float64,
    )
    return embeddings, quality, cost


def test_random_policy_is_reproducible_without_mutable_rng_state() -> None:
    embeddings, _, _ = training_data()
    policy = RandomPolicy(n_models=3, seed=42)

    first = policy.select(embeddings, quality_bias=0.5)
    second = policy.select(embeddings, quality_bias=0.5)

    np.testing.assert_array_equal(first, second)
    assert np.all((first >= 0) & (first < 3))


def test_constant_baselines_use_training_outcomes_only() -> None:
    _, quality, cost = training_data()

    cheapest, best_single = constant_baselines(quality, cost)

    assert cheapest.model_index == 1
    assert best_single.model_index == 0
    np.testing.assert_array_equal(cheapest.select(np.zeros((3, 2)), 1.0), [1, 1, 1])
    np.testing.assert_array_equal(best_single.select(np.zeros((3, 2)), 0.0), [0, 0, 0])


def test_global_utility_policy_moves_between_quality_and_cost_extremes() -> None:
    _, quality, cost = training_data()
    policy = GlobalUtilityPolicy.fit(quality, cost)
    inputs = np.zeros((2, 2), dtype=np.float32)

    quality_choices = policy.select(inputs, quality_bias=1.0)
    cost_choices = policy.select(inputs, quality_bias=0.0)

    np.testing.assert_array_equal(quality_choices, [0, 0])
    np.testing.assert_array_equal(cost_choices, [1, 1])


def test_ridge_policy_learns_prompt_dependent_model_quality() -> None:
    embeddings, quality, cost = training_data()
    policy = fit_ridge_policy(embeddings, quality, cost, alpha=0.01)

    choices = policy.select(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        quality_bias=1.0,
    )

    np.testing.assert_array_equal(choices, [0, 1])


def test_knn_policy_uses_only_supplied_training_rows() -> None:
    embeddings, quality, cost = training_data()
    policy = fit_knn_policy(embeddings[:2], quality[:2], cost[:2], neighbors=1)

    choices = policy.select(
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        quality_bias=1.0,
    )

    np.testing.assert_array_equal(choices, [0, 0])


def test_oracle_uses_each_held_out_outcome_row_as_an_upper_bound() -> None:
    quality = np.array([[0.9, 0.2], [0.1, 1.0]], dtype=np.float32)
    cost = np.array([[0.5, 0.1], [0.5, 0.1]], dtype=np.float64)

    quality_choices = oracle_select(quality, cost, quality_bias=1.0)
    cost_choices = oracle_select(quality, cost, quality_bias=0.0)

    np.testing.assert_array_equal(quality_choices, [0, 1])
    np.testing.assert_array_equal(cost_choices, [1, 1])
