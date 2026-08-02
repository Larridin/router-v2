from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from routerlab.weave import (
    WeaveConfig,
    WeavePolicy,
    fit_weave_policy,
    zscore_per_prompt,
)


def test_v075_config_matches_repository_metadata() -> None:
    config = WeaveConfig.v075()

    assert config.n_clusters == 16
    assert config.top_p == 4
    assert config.shrinkage == 10.0
    assert config.seed == 42
    assert config.n_init == 10


def test_prompt_quality_is_zscored_across_model_columns() -> None:
    quality = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])

    normalized = zscore_per_prompt(quality)

    np.testing.assert_allclose(normalized[0], [0.295875855, 0.5, 0.704124145])
    np.testing.assert_array_equal(normalized[1], [0.5, 0.5, 0.5])


def test_prompt_zscore_clips_before_mapping_to_unit_interval() -> None:
    quality = np.array([[1.0, *([0.0] * 12)]])

    normalized = zscore_per_prompt(quality)

    assert normalized[0, 0] == 1.0
    np.testing.assert_allclose(normalized[0, 1:], 0.4518874775675312)


def test_policy_moves_between_weave_quality_and_cost_endpoints() -> None:
    policy = WeavePolicy(
        models=("quality-model", "cheap-model"),
        centroids=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        cluster_quality=np.array([[0.8, -0.2], [-0.4, 0.6]], dtype=np.float64),
        expected_cost=np.array([0.9, 0.1], dtype=np.float64),
        config=WeaveConfig(
            n_clusters=2,
            top_p=1,
            shrinkage=2.0,
            seed=42,
            n_init=10,
        ),
    )
    prompts = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    np.testing.assert_array_equal(policy.select(prompts, quality_bias=1.0), [0, 1])
    np.testing.assert_array_equal(policy.select(prompts, quality_bias=0.0), [1, 1])


def test_top_p_clusters_are_summed_equally_and_ties_follow_model_order() -> None:
    policy = WeavePolicy(
        models=("first", "second"),
        centroids=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        cluster_quality=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        expected_cost=np.array([0.2, 0.2], dtype=np.float64),
        config=WeaveConfig(
            n_clusters=2,
            top_p=2,
            shrinkage=2.0,
            seed=42,
            n_init=10,
        ),
    )

    selected = policy.select(
        np.array([[1.0, 1.0]], dtype=np.float32),
        quality_bias=1.0,
    )

    np.testing.assert_array_equal(selected, [0])


def test_fit_uses_prompt_zscores_and_empirical_bayes_shrinkage() -> None:
    embeddings = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=np.float32,
    )
    quality = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    cost = np.tile([0.3, 0.1], (4, 1)).astype(np.float64)
    config = WeaveConfig(
        n_clusters=2,
        top_p=1,
        shrinkage=2.0,
        seed=42,
        n_init=10,
    )

    first = fit_weave_policy(embeddings, quality, cost, ("a", "b"), config=config)
    second = fit_weave_policy(embeddings, quality, cost, ("a", "b"), config=config)
    cluster_a = int(np.argmax(first.centroids[:, 0]))
    cluster_b = int(np.argmax(first.centroids[:, 1]))

    np.testing.assert_allclose(first.cluster_quality[cluster_a], [7 / 12, 5 / 12])
    np.testing.assert_allclose(first.cluster_quality[cluster_b], [5 / 12, 7 / 12])
    assert first.digest == second.digest
    np.testing.assert_array_equal(
        first.select(np.array([[1.0, 0.0], [0.0, 1.0]]), quality_bias=1.0),
        [0, 1],
    )


def test_fit_reassigns_training_rows_by_cosine_to_normalized_centroids() -> None:
    random = np.random.default_rng(3)
    embeddings = random.normal(size=(40, 3)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    quality = random.random((40, 2))
    cost = np.tile([0.3, 0.1], (40, 1))
    config = WeaveConfig(
        n_clusters=3,
        top_p=1,
        shrinkage=2.0,
        seed=42,
        n_init=10,
    )

    policy = fit_weave_policy(
        embeddings,
        quality,
        cost,
        ("a", "b"),
        config=config,
    )
    labels = np.argmax(embeddings @ policy.centroids.T, axis=1)
    normalized_quality = zscore_per_prompt(quality)
    global_mean = np.mean(normalized_quality, axis=0)
    expected = np.empty_like(policy.cluster_quality)
    for cluster in range(config.n_clusters):
        rows = normalized_quality[labels == cluster]
        expected[cluster] = (np.sum(rows, axis=0) + config.shrinkage * global_mean) / (
            len(rows) + config.shrinkage
        )

    np.testing.assert_allclose(policy.cluster_quality, expected)


def test_config_rejects_top_p_larger_than_cluster_count() -> None:
    with pytest.raises(ValueError, match="top_p"):
        WeaveConfig(
            n_clusters=2,
            top_p=3,
            shrinkage=10.0,
            seed=42,
            n_init=10,
        )


def test_fit_digest_is_reproducible_across_processes() -> None:
    script = """
import numpy as np
from routerlab.weave import WeaveConfig, fit_weave_policy

random = np.random.default_rng(42)
embeddings = random.normal(size=(1000, 64)).astype(np.float32)
quality = random.random((1000, 5))
cost = random.random((1000, 5))
config = WeaveConfig(16, 4, 10.0, 42, 10)
print(fit_weave_policy(embeddings, quality, cost, tuple("abcde"), config=config).digest)
"""
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment.pop(name, None)

    digests = [
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        ).stdout.strip()
        for _ in range(2)
    ]

    assert digests[0] == digests[1]
