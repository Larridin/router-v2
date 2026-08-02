from __future__ import annotations

import copy
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from routerlab.artifact import (
    EmbeddingSpec,
    load_artifact,
    require_split_seed,
    write_artifact,
)
from routerlab.centroid import CentroidConfig, CentroidPolicy


def policy() -> CentroidPolicy:
    return CentroidPolicy(
        models=("model-a", "model-b"),
        centroids=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        cluster_quality=np.array([[0.2, 0.9], [0.8, 0.3]], dtype=np.float32),
        expected_cost=np.array([0.1, 0.4], dtype=np.float64),
        config=CentroidConfig(
            n_clusters=2,
            top_p=1,
            shrinkage=5.0,
            temperature=0.05,
            seed=42,
        ),
    )


def embedding_spec() -> EmbeddingSpec:
    return EmbeddingSpec(
        model_id="test/embedder",
        source_repo="test/source",
        revision="abc123",
        dimension=2,
    )


def artifact_id(manifest: dict[str, Any]) -> str:
    identity = copy.deepcopy(
        {key: value for key, value in manifest.items() if key != "artifact_id"}
    )
    identity["provenance"] = {
        key: value for key, value in identity["provenance"].items() if key != "trained_at"
    }
    config = identity["config"]
    for field in ("shrinkage", "temperature"):
        value = float(config[field])
        config[field] = f"f64:{struct.pack('>d', value).hex()}"
    identity["expected_cost"] = [
        f"f64:{struct.pack('>d', float(value)).hex()}" for value in identity["expected_cost"]
    ]
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_artifact_round_trip_preserves_policy_decisions(tmp_path: Path) -> None:
    manifest = write_artifact(
        policy(),
        tmp_path / "artifact",
        embedding=embedding_spec(),
        provenance={"dataset_sha256": "deadbeef"},
        trained_at="2026-08-01T12:00:00Z",
    )

    loaded = load_artifact(tmp_path / "artifact")
    original_decision = policy().route_vector(np.array([1.0, 0.0]), quality_bias=0.75)
    loaded_decision = loaded.policy.route_vector(np.array([1.0, 0.0]), quality_bias=0.75)

    assert loaded.manifest == manifest
    assert loaded_decision == original_decision
    np.testing.assert_array_equal(loaded.policy.centroids, policy().centroids)
    np.testing.assert_array_equal(loaded.policy.cluster_quality, policy().cluster_quality)
    np.testing.assert_array_equal(loaded.policy.expected_cost, policy().expected_cost)


def test_training_timestamp_does_not_change_artifact_identity(tmp_path: Path) -> None:
    first = write_artifact(
        policy(),
        tmp_path / "first",
        embedding=embedding_spec(),
        provenance={"source": "fixture"},
        trained_at="2026-08-01T12:00:00Z",
    )
    second = write_artifact(
        policy(),
        tmp_path / "second",
        embedding=embedding_spec(),
        provenance={"source": "fixture"},
        trained_at="2027-01-01T00:00:00Z",
    )

    assert first.artifact_id == second.artifact_id
    assert first.provenance != second.provenance


def test_artifact_identity_binds_stable_training_provenance(tmp_path: Path) -> None:
    first_directory = tmp_path / "first"
    first = write_artifact(
        policy(),
        first_directory,
        embedding=embedding_spec(),
        provenance={"training_matrix_sha256": "a" * 64, "split_seed": 42},
        trained_at="2026-08-01T12:00:00Z",
    )
    second = write_artifact(
        policy(),
        tmp_path / "second",
        embedding=embedding_spec(),
        provenance={"training_matrix_sha256": "b" * 64, "split_seed": 42},
        trained_at="2026-08-01T12:00:00Z",
    )

    assert first.artifact_id != second.artifact_id
    manifest_path = first_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["split_seed"] = 43
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_id"):
        load_artifact(first_directory)


def test_artifact_split_seed_must_be_present_and_match(tmp_path: Path) -> None:
    matching = write_artifact(
        policy(),
        tmp_path / "matching",
        embedding=embedding_spec(),
        provenance={"split_seed": 42},
        trained_at="2026-08-01T12:00:00Z",
    )
    missing = write_artifact(
        policy(),
        tmp_path / "missing",
        embedding=embedding_spec(),
        trained_at="2026-08-01T12:00:00Z",
    )

    require_split_seed(matching, 42)
    with pytest.raises(ValueError, match="does not match"):
        require_split_seed(matching, 43)
    with pytest.raises(ValueError, match="no valid split seed"):
        require_split_seed(missing, 42)


@pytest.mark.parametrize("tensor", ["centroids.f32", "cluster_quality.f32"])
def test_artifact_rejects_tensor_corruption(tmp_path: Path, tensor: str) -> None:
    directory = tmp_path / "artifact"
    write_artifact(
        policy(), directory, embedding=embedding_spec(), trained_at="2026-08-01T12:00:00Z"
    )
    path = directory / tensor
    payload = bytearray(path.read_bytes())
    payload[0] ^= 0xFF
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_artifact(directory)


def test_artifact_rejects_correctly_hashed_non_unit_centroid(tmp_path: Path) -> None:
    directory = tmp_path / "artifact"
    write_artifact(
        policy(), directory, embedding=embedding_spec(), trained_at="2026-08-01T12:00:00Z"
    )
    tensor = np.array([[2.0, 0.0], [0.0, 1.0]], dtype="<f4").tobytes()
    (directory / "centroids.f32").write_bytes(tensor)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tensors"]["centroids"]["sha256"] = hashlib.sha256(tensor).hexdigest()
    manifest["artifact_id"] = artifact_id(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unit length"):
        load_artifact(directory)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema="future-schema"), "unknown artifact schema"),
        (lambda value: value.update(unknown=True), "manifest has unknown fields"),
        (
            lambda value: value["embedding"].update(unknown=True),
            "embedding has unknown fields",
        ),
        (
            lambda value: value["config"].update(unknown=True),
            "config has unknown fields",
        ),
        (
            lambda value: value["tensors"]["centroids"].update(unknown=True),
            "centroids has unknown fields",
        ),
        (
            lambda value: value.update(models=["model-a", "model-a"]),
            "models must be non-empty and unique",
        ),
        (
            lambda value: value["tensors"]["centroids"].update(shape=[2, 3]),
            "centroids shape",
        ),
        (
            lambda value: value.update(expected_cost=[float("nan"), 0.4]),
            "expected_cost must be finite",
        ),
    ],
)
def test_artifact_rejects_invalid_manifest(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    directory = tmp_path / "artifact"
    write_artifact(
        policy(), directory, embedding=embedding_spec(), trained_at="2026-08-01T12:00:00Z"
    )
    path = directory / "manifest.json"
    value = json.loads(path.read_text())
    mutation(value)  # type: ignore[operator]
    path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match=message):
        load_artifact(directory)
