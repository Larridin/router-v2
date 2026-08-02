"""Immutable value types shared by training and evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Float32Matrix = NDArray[np.float32]
Float64Matrix = NDArray[np.float64]
Int64Matrix = NDArray[np.int64]


class _Hash(Protocol):
    def update(self, value: bytes) -> None: ...


@dataclass(frozen=True, slots=True)
class OutcomeMatrix:
    """Prompt-by-model full-information outcomes with aligned metadata."""

    prompt_keys: tuple[str, ...]
    datasets: tuple[str, ...]
    source_indices: tuple[str, ...]
    prompts: tuple[str, ...]
    models: tuple[str, ...]
    quality: Float32Matrix
    realized_cost: Float64Matrix
    prompt_tokens: Int64Matrix
    completion_tokens: Int64Matrix

    def __post_init__(self) -> None:
        rows = len(self.prompt_keys)
        columns = len(self.models)
        for name, values in (
            ("datasets", self.datasets),
            ("source_indices", self.source_indices),
            ("prompts", self.prompts),
        ):
            if len(values) != rows:
                raise ValueError(f"{name} has {len(values)} rows, expected {rows}")
        if len(set(self.prompt_keys)) != rows:
            raise ValueError("prompt_keys must be unique")
        if not self.models or len(set(self.models)) != columns:
            raise ValueError("models must be non-empty and unique")
        expected = (rows, columns)
        for name, values in (
            ("quality", self.quality),
            ("realized_cost", self.realized_cost),
            ("prompt_tokens", self.prompt_tokens),
            ("completion_tokens", self.completion_tokens),
        ):
            if values.shape != expected:
                raise ValueError(f"{name} shape {values.shape}, expected {expected}")
        if not np.all(np.isfinite(self.quality)):
            raise ValueError("quality contains non-finite values")
        if not np.all(np.isfinite(self.realized_cost)):
            raise ValueError("realized_cost contains non-finite values")
        if np.any(self.realized_cost < 0):
            raise ValueError("realized_cost contains negative values")
        if np.any(self.prompt_tokens < 0) or np.any(self.completion_tokens < 0):
            raise ValueError("token counts must be non-negative")

    @property
    def n_prompts(self) -> int:
        """Return the number of aligned prompts."""

        return len(self.prompt_keys)

    @property
    def n_models(self) -> int:
        """Return the number of models in the shared roster."""

        return len(self.models)

    def take(self, indices: NDArray[np.integer] | NDArray[np.bool_]) -> OutcomeMatrix:
        """Return rows selected by integer indices or a boolean mask."""

        selected = np.arange(self.n_prompts)[indices]
        return OutcomeMatrix(
            prompt_keys=tuple(self.prompt_keys[index] for index in selected),
            datasets=tuple(self.datasets[index] for index in selected),
            source_indices=tuple(self.source_indices[index] for index in selected),
            prompts=tuple(self.prompts[index] for index in selected),
            models=self.models,
            quality=self.quality[selected].copy(),
            realized_cost=self.realized_cost[selected].copy(),
            prompt_tokens=self.prompt_tokens[selected].copy(),
            completion_tokens=self.completion_tokens[selected].copy(),
        )


@dataclass(frozen=True, slots=True)
class LoadAudit:
    """Counts explaining how raw records became a complete matrix."""

    files: int
    raw_records: int
    accepted_prompts: int
    rejections: Mapping[str, int]
    normalizations: Mapping[str, int] = field(default_factory=dict)


def outcome_matrix_digest(matrix: OutcomeMatrix) -> str:
    """Return a canonical SHA-256 digest of matrix metadata and outcomes."""

    digest = hashlib.sha256(b"routerlab-outcome-matrix-v1\0")
    for name, values in (
        ("prompt_keys", matrix.prompt_keys),
        ("datasets", matrix.datasets),
        ("source_indices", matrix.source_indices),
        ("prompts", matrix.prompts),
        ("models", matrix.models),
    ):
        _update_text_sequence(digest, name, values)
    for name, values, dtype in (
        ("quality", matrix.quality, "<f4"),
        ("realized_cost", matrix.realized_cost, "<f8"),
        ("prompt_tokens", matrix.prompt_tokens, "<i8"),
        ("completion_tokens", matrix.completion_tokens, "<i8"),
    ):
        encoded_name = name.encode("ascii")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        array = np.ascontiguousarray(values, dtype=dtype)
        digest.update(array.ndim.to_bytes(4, "big"))
        for dimension in array.shape:
            digest.update(dimension.to_bytes(8, "big"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _update_text_sequence(
    digest: _Hash,
    name: str,
    values: tuple[str, ...],
) -> None:
    encoded_name = name.encode("ascii")
    digest.update(len(encoded_name).to_bytes(4, "big"))
    digest.update(encoded_name)
    digest.update(len(values).to_bytes(8, "big"))
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
