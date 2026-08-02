from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from routerlab.artifact import EmbeddingSpec, write_artifact
from routerlab.centroid import CentroidConfig, CentroidPolicy

ROOT = Path(__file__).parents[1]
RUST = ROOT / "rust"


def test_python_artifact_and_rust_scorer_match_decisions(tmp_path: Path) -> None:
    policy = CentroidPolicy(
        models=("model-a", "model-b", "model-c"),
        centroids=np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        cluster_quality=np.array(
            [[0.2, 0.9, 0.5], [0.8, 0.3, 0.4], [0.5, 0.6, 0.95]],
            dtype=np.float32,
        ),
        expected_cost=np.array([0.1, 0.4, 0.2], dtype=np.float64),
        config=CentroidConfig(3, 2, 5.0, 0.1, 42),
    )
    artifact = tmp_path / "artifact"
    write_artifact(
        policy,
        artifact,
        embedding=EmbeddingSpec("test/embedder", "test/source", "v1", 3),
        trained_at="2026-08-01T12:00:00Z",
    )
    subprocess.run(
        ["cargo", "build", "--quiet"],
        cwd=RUST,
        check=True,
    )
    binary = RUST / "target" / "debug" / "routerlab-core"
    random = np.random.default_rng(42)

    for vector in random.normal(size=(20, 3)).astype(np.float32):
        for bias in (0.0, 0.25, 0.5, 0.75, 1.0):
            expected = policy.route_vector(vector, quality_bias=bias)
            process = subprocess.run(
                [binary, "--artifact", artifact],
                input=json.dumps({"embedding": vector.tolist(), "quality_bias": bias}),
                capture_output=True,
                text=True,
                check=True,
            )
            actual = json.loads(process.stdout)

            assert actual["model"] == expected.model
            assert actual["model_index"] == expected.model_index
            assert actual["cluster_ids"] == list(expected.cluster_ids)
            np.testing.assert_allclose(
                actual["cluster_weights"], expected.cluster_weights, rtol=1e-6, atol=1e-8
            )
            np.testing.assert_allclose(
                actual["predicted_quality"], expected.predicted_quality, rtol=1e-6
            )
