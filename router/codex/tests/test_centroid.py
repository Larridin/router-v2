from __future__ import annotations

import numpy as np
import pytest

from routerlab.centroid import (
    CentroidConfig,
    CentroidPolicy,
    fit_centroid_policy,
    shrink_cluster_quality,
)


def example_policy() -> CentroidPolicy:
    return CentroidPolicy(
        models=("model-a", "model-b", "model-c"),
        centroids=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        cluster_quality=np.array([[0.2, 0.9, 0.5], [0.8, 0.3, 0.5]], dtype=np.float32),
        expected_cost=np.array([0.1, 0.3, 0.2], dtype=np.float64),
        config=CentroidConfig(
            n_clusters=2,
            top_p=1,
            shrinkage=5.0,
            temperature=0.05,
            seed=42,
        ),
    )


def test_shrinkage_uses_cluster_evidence_and_global_prior() -> None:
    quality = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1], dtype=np.int64)

    means = shrink_cluster_quality(labels, quality, n_clusters=3, shrinkage=2.0)

    global_mean = np.array([2 / 3, 2 / 3], dtype=np.float64)
    np.testing.assert_allclose(means[0], ([1.0, 1.0] + 2 * global_mean) / 4)
    np.testing.assert_allclose(means[1], ([1.0, 1.0] + 2 * global_mean) / 3)
    np.testing.assert_allclose(means[2], global_mean)


def test_route_vector_respects_quality_and_cost_endpoints() -> None:
    policy = example_policy()

    quality = policy.route_vector(np.array([1.0, 0.0], dtype=np.float32), quality_bias=1.0)
    cost = policy.route_vector(np.array([1.0, 0.0], dtype=np.float32), quality_bias=0.0)

    assert quality.model == "model-b"
    assert quality.cluster_ids == (0,)
    assert quality.predicted_quality == pytest.approx(0.9)
    assert cost.model == "model-a"
    assert cost.expected_cost == pytest.approx(0.1)


def test_route_vector_filters_eligibility_before_normalizing() -> None:
    decision = example_policy().route_vector(
        np.array([1.0, 0.0], dtype=np.float32),
        eligible_models=("model-a", "model-c"),
        quality_bias=1.0,
    )

    assert decision.model == "model-c"


def test_route_vector_uses_manifest_order_for_exact_ties() -> None:
    policy = CentroidPolicy(
        models=("first", "second"),
        centroids=np.array([[1.0, 0.0]], dtype=np.float32),
        cluster_quality=np.array([[0.5, 0.5]], dtype=np.float32),
        expected_cost=np.array([0.2, 0.2], dtype=np.float64),
        config=CentroidConfig(1, 1, 1.0, 0.1, 42),
    )

    decision = policy.route_vector(np.array([1.0, 0.0]), quality_bias=0.5)

    assert decision.model == "first"


def test_route_vector_blends_top_p_clusters_with_softmax() -> None:
    policy = example_policy()
    policy = CentroidPolicy(
        models=policy.models,
        centroids=policy.centroids,
        cluster_quality=policy.cluster_quality,
        expected_cost=policy.expected_cost,
        config=CentroidConfig(2, 2, 5.0, 1.0, 42),
    )

    decision = policy.route_vector(np.array([1.0, 1.0]), quality_bias=1.0)

    assert decision.cluster_ids == (0, 1)
    assert decision.model == "model-b"
    assert decision.predicted_quality == pytest.approx(0.6)


@pytest.mark.parametrize("bias", [-0.1, 1.1, float("nan")])
def test_route_vector_rejects_invalid_bias(bias: float) -> None:
    with pytest.raises(ValueError, match="quality_bias"):
        example_policy().route_vector(np.array([1.0, 0.0]), quality_bias=bias)


def test_fit_is_deterministic_for_fixed_seed() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.02, 0.98],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    quality = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.1, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
        ],
        dtype=np.float32,
    )
    cost = np.tile(np.array([[0.1, 0.2]], dtype=np.float64), (6, 1))
    config = CentroidConfig(2, 1, 2.0, 0.05, 42)

    first = fit_centroid_policy(embeddings, quality, cost, ("a", "b"), config)
    second = fit_centroid_policy(embeddings, quality, cost, ("a", "b"), config)

    np.testing.assert_array_equal(first.centroids, second.centroids)
    np.testing.assert_array_equal(first.cluster_quality, second.cluster_quality)
    np.testing.assert_array_equal(first.expected_cost, second.expected_cost)
