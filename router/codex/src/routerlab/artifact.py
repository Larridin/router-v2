"""Content-addressed serialization for centroid routing policies."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from routerlab.centroid import CentroidConfig, CentroidPolicy

SCHEMA = "codex-centroid-artifact-v2"
MANIFEST_FILE = "manifest.json"
_HASH = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class EmbeddingSpec:
    """Embedding identity required to reproduce prompt vectors."""

    model_id: str
    source_repo: str
    revision: str
    dimension: int

    def __post_init__(self) -> None:
        if not self.model_id or not self.source_repo or not self.revision:
            raise ValueError("embedding model_id, source_repo, and revision must not be empty")
        if self.dimension <= 0:
            raise ValueError("embedding dimension must be positive")


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Location, shape, and digest of one little-endian float tensor."""

    file: str
    shape: tuple[int, int]
    sha256: str


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Validated routing artifact metadata."""

    schema: str
    artifact_id: str
    models: tuple[str, ...]
    embedding: EmbeddingSpec
    config: CentroidConfig
    expected_cost: tuple[float, ...]
    tensors: dict[str, TensorSpec]
    provenance: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LoadedArtifact:
    """A manifest and its reconstructed in-memory policy."""

    manifest: ArtifactManifest
    policy: CentroidPolicy


def require_split_seed(manifest: ArtifactManifest, expected_seed: int) -> None:
    """Reject an artifact whose training split cannot match evaluation."""

    artifact_seed = manifest.provenance.get("split_seed")
    if not isinstance(artifact_seed, int) or isinstance(artifact_seed, bool):
        raise ValueError("artifact provenance has no valid split seed")
    if expected_seed != artifact_seed:
        raise ValueError(
            f"evaluation split seed {expected_seed} does not match artifact split seed "
            f"{artifact_seed}"
        )


def require_evaluation_context(
    manifest: ArtifactManifest,
    *,
    split_seed: int,
    embedding: EmbeddingSpec,
    training_matrix_sha256: str,
) -> None:
    """Require evaluation inputs to match the artifact's training context."""

    require_split_seed(manifest, split_seed)
    if manifest.embedding != embedding:
        raise ValueError("evaluation embedding identity does not match artifact")
    artifact_matrix = manifest.provenance.get("training_matrix_sha256")
    if not isinstance(artifact_matrix, str) or _HASH.fullmatch(artifact_matrix) is None:
        raise ValueError("artifact provenance has no valid training matrix SHA-256")
    if artifact_matrix != training_matrix_sha256:
        raise ValueError("evaluation training matrix does not match artifact provenance")


def write_artifact(
    policy: CentroidPolicy,
    directory: Path,
    *,
    embedding: EmbeddingSpec,
    trained_at: str,
    provenance: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """Write an immutable manifest and little-endian policy tensors."""

    if embedding.dimension != policy.dimension:
        raise ValueError("embedding dimension does not match policy")
    if not trained_at:
        raise ValueError("trained_at must not be empty")
    output = Path(directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"artifact directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, TensorSpec] = {}
    for name, values in (
        ("centroids", policy.centroids),
        ("cluster_quality", policy.cluster_quality),
    ):
        filename = f"{name}.f32"
        payload = np.asarray(values, dtype="<f4", order="C").tobytes(order="C")
        (output / filename).write_bytes(payload)
        tensors[name] = TensorSpec(
            file=filename,
            shape=(int(values.shape[0]), int(values.shape[1])),
            sha256=hashlib.sha256(payload).hexdigest(),
        )

    identity_provenance = dict(provenance or {})
    if "trained_at" in identity_provenance:
        raise ValueError("trained_at is supplied separately")
    _canonical_json(identity_provenance)
    _validate_identity_provenance(identity_provenance)
    provenance_values = dict(identity_provenance)
    provenance_values["trained_at"] = trained_at
    identity = _identity_dict(
        models=policy.models,
        embedding=embedding,
        config=policy.config,
        expected_cost=tuple(float(value) for value in policy.expected_cost),
        tensors=tensors,
        provenance=identity_provenance,
        encode_floats=True,
    )
    artifact_id = hashlib.sha256(_canonical_json(identity)).hexdigest()
    manifest = ArtifactManifest(
        schema=SCHEMA,
        artifact_id=artifact_id,
        models=policy.models,
        embedding=embedding,
        config=policy.config,
        expected_cost=tuple(float(value) for value in policy.expected_cost),
        tensors=tensors,
        provenance=provenance_values,
    )
    (output / MANIFEST_FILE).write_bytes(_canonical_json(_manifest_dict(manifest)) + b"\n")
    return manifest


def load_artifact(directory: Path) -> LoadedArtifact:
    """Validate and reconstruct a centroid policy from an artifact directory."""

    root = Path(directory)
    try:
        raw = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read artifact manifest: {error}") from error
    manifest = _parse_manifest(raw)
    expected_id = hashlib.sha256(
        _canonical_json(
            _identity_dict(
                models=manifest.models,
                embedding=manifest.embedding,
                config=manifest.config,
                expected_cost=manifest.expected_cost,
                tensors=manifest.tensors,
                provenance=_identity_provenance(manifest.provenance),
                encode_floats=True,
            )
        )
    ).hexdigest()
    if manifest.artifact_id != expected_id:
        raise ValueError("artifact_id does not match manifest content")

    centroids = _read_tensor(root, manifest.tensors["centroids"])
    cluster_quality = _read_tensor(root, manifest.tensors["cluster_quality"])
    centroid_norms = np.linalg.norm(centroids, axis=1)
    if np.any(np.abs(centroid_norms - 1.0) > 1e-5):
        raise ValueError("centroid rows must have unit length")
    policy = CentroidPolicy(
        models=manifest.models,
        centroids=centroids,
        cluster_quality=cluster_quality,
        expected_cost=np.asarray(manifest.expected_cost, dtype=np.float64),
        config=manifest.config,
    )
    return LoadedArtifact(manifest=manifest, policy=policy)


def _parse_manifest(raw: object) -> ArtifactManifest:
    if not isinstance(raw, dict):
        raise ValueError("artifact manifest must be an object")
    _require_exact_keys(
        raw,
        "manifest",
        {
            "schema",
            "artifact_id",
            "models",
            "embedding",
            "config",
            "expected_cost",
            "tensors",
            "provenance",
        },
    )
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"unknown artifact schema: {raw.get('schema')!r}")

    models_raw = raw.get("models")
    if not isinstance(models_raw, list) or not all(
        isinstance(model, str) and model for model in models_raw
    ):
        raise ValueError("models must be non-empty and unique")
    models = tuple(models_raw)
    if not models or len(set(models)) != len(models):
        raise ValueError("models must be non-empty and unique")

    embedding_raw = _object(raw.get("embedding"), "embedding")
    _require_exact_keys(
        embedding_raw,
        "embedding",
        {"model_id", "source_repo", "revision", "dimension"},
    )
    embedding = EmbeddingSpec(
        model_id=_string(embedding_raw.get("model_id"), "embedding model_id"),
        source_repo=_string(embedding_raw.get("source_repo"), "embedding source_repo"),
        revision=_string(embedding_raw.get("revision"), "embedding revision"),
        dimension=_positive_int(embedding_raw.get("dimension"), "embedding dimension"),
    )
    config_raw = _object(raw.get("config"), "config")
    _require_exact_keys(
        config_raw,
        "config",
        {"n_clusters", "top_p", "shrinkage", "temperature", "seed"},
    )
    try:
        config = CentroidConfig(
            n_clusters=_positive_int(config_raw.get("n_clusters"), "n_clusters"),
            top_p=_positive_int(config_raw.get("top_p"), "top_p"),
            shrinkage=float(config_raw.get("shrinkage")),
            temperature=float(config_raw.get("temperature")),
            seed=_non_negative_int(config_raw.get("seed"), "seed"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid centroid config: {error}") from error

    expected_raw = raw.get("expected_cost")
    if not isinstance(expected_raw, list) or len(expected_raw) != len(models):
        raise ValueError("expected_cost must match models")
    try:
        expected_cost = tuple(float(value) for value in expected_raw)
    except (TypeError, ValueError) as error:
        raise ValueError("expected_cost must be numeric") from error
    if not all(math.isfinite(value) for value in expected_cost):
        raise ValueError("expected_cost must be finite")
    if any(value < 0 for value in expected_cost):
        raise ValueError("expected_cost must be non-negative")

    tensors_raw = _object(raw.get("tensors"), "tensors")
    if set(tensors_raw) != {"centroids", "cluster_quality"}:
        raise ValueError("tensors must contain centroids and cluster_quality")
    tensors = {
        name: _parse_tensor(name, tensors_raw[name]) for name in ("centroids", "cluster_quality")
    }
    expected_shapes = {
        "centroids": (config.n_clusters, embedding.dimension),
        "cluster_quality": (config.n_clusters, len(models)),
    }
    for name, shape in expected_shapes.items():
        if tensors[name].shape != shape:
            raise ValueError(f"{name} shape {tensors[name].shape}, expected {shape}")

    artifact_id = _string(raw.get("artifact_id"), "artifact_id")
    if _HASH.fullmatch(artifact_id) is None:
        raise ValueError("artifact_id must be a lowercase SHA-256 digest")
    provenance = _object(raw.get("provenance"), "provenance")
    _string(provenance.get("trained_at"), "trained_at")
    _canonical_json(provenance)
    return ArtifactManifest(
        schema=SCHEMA,
        artifact_id=artifact_id,
        models=models,
        embedding=embedding,
        config=config,
        expected_cost=expected_cost,
        tensors=tensors,
        provenance=provenance,
    )


def _parse_tensor(name: str, raw: object) -> TensorSpec:
    value = _object(raw, name)
    _require_exact_keys(value, name, {"file", "shape", "sha256"})
    filename = _string(value.get("file"), f"{name} file")
    if filename != f"{name}.f32":
        raise ValueError(f"invalid {name} filename")
    shape_raw = value.get("shape")
    if not isinstance(shape_raw, list) or len(shape_raw) != 2:
        raise ValueError(f"{name} shape must have two dimensions")
    shape = tuple(_positive_int(item, f"{name} shape") for item in shape_raw)
    digest = _string(value.get("sha256"), f"{name} sha256")
    if _HASH.fullmatch(digest) is None:
        raise ValueError(f"{name} sha256 must be a lowercase digest")
    return TensorSpec(file=filename, shape=(shape[0], shape[1]), sha256=digest)


def _read_tensor(root: Path, spec: TensorSpec) -> NDArray[np.float32]:
    try:
        payload = (root / spec.file).read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read tensor {spec.file}: {error}") from error
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != spec.sha256:
        raise ValueError(f"{spec.file} SHA-256 mismatch")
    expected_bytes = math.prod(spec.shape) * np.dtype("<f4").itemsize
    if len(payload) != expected_bytes:
        raise ValueError(f"{spec.file} byte length does not match shape")
    values = np.frombuffer(payload, dtype="<f4").reshape(spec.shape).astype(np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{spec.file} contains non-finite values")
    return values


def _identity_dict(
    *,
    models: tuple[str, ...],
    embedding: EmbeddingSpec,
    config: CentroidConfig,
    expected_cost: tuple[float, ...],
    tensors: dict[str, TensorSpec],
    provenance: dict[str, Any],
    encode_floats: bool,
) -> dict[str, Any]:
    number = _identity_float if encode_floats else float
    return {
        "schema": SCHEMA,
        "models": list(models),
        "embedding": {
            "model_id": embedding.model_id,
            "source_repo": embedding.source_repo,
            "revision": embedding.revision,
            "dimension": embedding.dimension,
        },
        "config": {
            "n_clusters": config.n_clusters,
            "top_p": config.top_p,
            "shrinkage": number(config.shrinkage),
            "temperature": number(config.temperature),
            "seed": config.seed,
        },
        "expected_cost": [number(value) for value in expected_cost],
        "tensors": {
            name: {
                "file": spec.file,
                "shape": list(spec.shape),
                "sha256": spec.sha256,
            }
            for name, spec in sorted(tensors.items())
        },
        "provenance": provenance,
    }


def _manifest_dict(manifest: ArtifactManifest) -> dict[str, Any]:
    value = _identity_dict(
        models=manifest.models,
        embedding=manifest.embedding,
        config=manifest.config,
        expected_cost=manifest.expected_cost,
        tensors=manifest.tensors,
        provenance=_identity_provenance(manifest.provenance),
        encode_floats=False,
    )
    value["artifact_id"] = manifest.artifact_id
    value["provenance"] = manifest.provenance
    return value


def _identity_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    values = dict(provenance)
    trained_at = values.pop("trained_at", None)
    if not isinstance(trained_at, str) or not trained_at:
        raise ValueError("trained_at must be a non-empty string")
    _validate_identity_provenance(values)
    return values


def _identity_float(value: float) -> str:
    return f"f64:{struct.pack('>d', value).hex()}"


def _validate_identity_provenance(value: object) -> None:
    if isinstance(value, float):
        raise ValueError("artifact provenance floats must be encoded as strings")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("artifact provenance keys must be strings")
            _validate_identity_provenance(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _validate_identity_provenance(item)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"value is not canonical JSON: {error}") from error


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _require_exact_keys(value: dict[str, Any], name: str, expected: set[str]) -> None:
    unknown = sorted(set(value) - expected)
    if unknown:
        raise ValueError(f"{name} has unknown fields: {', '.join(unknown)}")
    missing = sorted(expected - set(value))
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(missing)}")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
