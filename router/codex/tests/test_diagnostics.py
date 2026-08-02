from __future__ import annotations

import hashlib

import numpy as np

from routerlab.centroid import CentroidConfig, CentroidPolicy
from routerlab.diagnostics import (
    measure_scorer_latency,
    perturb_prompt,
    robustness_diagnostics,
)


class SemanticFakeEncoder:
    model_id = "test/embedder"
    source_repo = "test/source"
    revision = "v1"
    dimension = 2

    def encode(self, texts: tuple[str, ...]) -> np.ndarray:
        rows = []
        for text in texts:
            normalized = text.lower()
            rows.append([1.0, 0.0] if "alpha" in normalized else [0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)


def policy() -> CentroidPolicy:
    return CentroidPolicy(
        models=("model-a", "model-b"),
        centroids=np.eye(2, dtype=np.float32),
        cluster_quality=np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32),
        expected_cost=np.array([0.2, 0.2]),
        config=CentroidConfig(2, 1, 1.0, 0.05, 42),
    )


def test_perturbations_are_deterministic_and_nonempty() -> None:
    prompt = "Alpha question"

    values = [
        perturb_prompt(prompt, kind) for kind in ("whitespace", "case", "punctuation", "wrapper")
    ]

    assert len(set(values)) == 4
    assert all(value and value != prompt for value in values)
    assert values == [
        perturb_prompt(prompt, kind) for kind in ("whitespace", "case", "punctuation", "wrapper")
    ]


def test_semantic_robustness_preserves_fake_router_decisions() -> None:
    prompts = tuple("Alpha request" if index % 2 == 0 else "Beta request" for index in range(20))
    keys = tuple(hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts)
    encoder = SemanticFakeEncoder()
    embeddings = encoder.encode(prompts)
    quality = np.array(
        [[1.0, 0.0] if index % 2 == 0 else [0.0, 1.0] for index in range(20)],
        dtype=np.float32,
    )
    cost = np.full((20, 2), 0.2)

    result = robustness_diagnostics(
        prompts,
        keys,
        embeddings,
        quality,
        cost,
        policy(),
        encoder,
        quality_biases=(0.75, 1.0),
        sample_size=12,
        seed=42,
    )

    assert result["sample_size"] == 12
    assert result["overall_flip_rate"] == 0.0
    assert result["mean_quality_delta"] == 0.0
    assert result["mean_cost_delta"] == 0.0


def test_scorer_latency_measures_every_route() -> None:
    embeddings = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (10, 1))

    result = measure_scorer_latency(policy(), embeddings, quality_biases=(0.5,), repeats=2)

    assert result["routes"] == 20
    assert result["p50_microseconds"] > 0
    assert result["p95_microseconds"] >= result["p50_microseconds"]
    assert result["p99_microseconds"] >= result["p95_microseconds"]
