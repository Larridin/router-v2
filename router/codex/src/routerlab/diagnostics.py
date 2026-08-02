"""Semantic perturbation stability and pure scorer latency diagnostics."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from routerlab.centroid import CentroidPolicy
from routerlab.embeddings import Encoder
from routerlab.metrics import selected_outcomes

PERTURBATIONS = ("whitespace", "case", "punctuation", "wrapper")


def perturb_prompt(prompt: str, kind: str) -> str:
    """Apply one deterministic, meaning-preserving surface perturbation."""

    if not prompt:
        raise ValueError("prompt must not be empty")
    if kind == "whitespace":
        return f" \n{prompt}\n "
    if kind == "case":
        for index, character in enumerate(prompt):
            if character.isalpha():
                return f"{prompt[:index]}{character.swapcase()}{prompt[index + 1 :]}"
        return f"{prompt} A"
    if kind == "punctuation":
        return f"{prompt}."
    if kind == "wrapper":
        return f"Please answer:\n{prompt}"
    raise ValueError(f"unknown perturbation: {kind}")


def robustness_diagnostics(
    prompts: tuple[str, ...],
    prompt_keys: tuple[str, ...],
    embeddings: NDArray[np.floating],
    quality: NDArray[np.floating],
    cost: NDArray[np.floating],
    policy: CentroidPolicy,
    encoder: Encoder,
    *,
    quality_biases: tuple[float, ...] = (0.75, 1.0),
    sample_size: int = 128,
    seed: int = 42,
) -> dict[str, Any]:
    """Measure decision and realized-outcome changes under prompt perturbations."""

    if len(prompts) != len(prompt_keys):
        raise ValueError("prompts and prompt_keys must align")
    vectors = np.asarray(embeddings, dtype=np.float32)
    quality_values = np.asarray(quality, dtype=np.float64)
    cost_values = np.asarray(cost, dtype=np.float64)
    expected_outcomes = (len(prompts), len(policy.models))
    if vectors.shape != (len(prompts), policy.dimension):
        raise ValueError("embeddings do not align with prompts and policy")
    if quality_values.shape != expected_outcomes or cost_values.shape != expected_outcomes:
        raise ValueError("outcomes do not align with prompts and policy models")
    if encoder.dimension != policy.dimension:
        raise ValueError("encoder dimension does not match policy")
    if not quality_biases or any(
        not math.isfinite(bias) or not 0 <= bias <= 1 for bias in quality_biases
    ):
        raise ValueError("quality_biases must be finite and in [0, 1]")
    if sample_size <= 0 or not prompts:
        raise ValueError("sample_size and prompts must be non-empty")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    indices = _sample_indices(prompt_keys, min(sample_size, len(prompts)), seed)
    sample_prompts = tuple(prompts[index] for index in indices)
    sample_vectors = vectors[indices]
    sample_quality = quality_values[indices]
    sample_cost = cost_values[indices]
    perturbed_prompts = tuple(
        perturb_prompt(prompt, kind) for kind in PERTURBATIONS for prompt in sample_prompts
    )
    perturbed_vectors = np.asarray(encoder.encode(perturbed_prompts), dtype=np.float32)
    expected_vectors = (len(PERTURBATIONS) * len(indices), policy.dimension)
    if perturbed_vectors.shape != expected_vectors or not np.all(np.isfinite(perturbed_vectors)):
        raise ValueError("encoder returned invalid perturbed embeddings")
    perturbed_vectors = perturbed_vectors.reshape(
        len(PERTURBATIONS), len(indices), policy.dimension
    )

    rows: list[dict[str, Any]] = []
    all_flips: list[NDArray[np.bool_]] = []
    all_quality_deltas: list[NDArray[np.float64]] = []
    all_cost_deltas: list[NDArray[np.float64]] = []
    for bias in quality_biases:
        original = policy.select_many(sample_vectors, bias)
        original_quality, original_cost = selected_outcomes(sample_quality, sample_cost, original)
        for perturbation_index, kind in enumerate(PERTURBATIONS):
            changed = policy.select_many(perturbed_vectors[perturbation_index], bias)
            changed_quality, changed_cost = selected_outcomes(sample_quality, sample_cost, changed)
            flips = changed != original
            quality_delta = changed_quality - original_quality
            cost_delta = changed_cost - original_cost
            all_flips.append(flips)
            all_quality_deltas.append(quality_delta)
            all_cost_deltas.append(cost_delta)
            rows.append(
                {
                    "quality_bias": bias,
                    "perturbation": kind,
                    "flip_rate": float(np.mean(flips)),
                    "mean_quality_delta": float(np.mean(quality_delta)),
                    "mean_cost_delta": float(np.mean(cost_delta)),
                }
            )
    return {
        "schema": "routerlab-robustness-v1",
        "seed": seed,
        "sample_size": len(indices),
        "quality_biases": list(quality_biases),
        "perturbations": list(PERTURBATIONS),
        "overall_flip_rate": float(np.mean(np.concatenate(all_flips))),
        "mean_quality_delta": float(np.mean(np.concatenate(all_quality_deltas))),
        "mean_cost_delta": float(np.mean(np.concatenate(all_cost_deltas))),
        "details": rows,
    }


def measure_scorer_latency(
    policy: CentroidPolicy,
    embeddings: NDArray[np.floating],
    *,
    quality_biases: tuple[float, ...] = (0.75, 1.0),
    repeats: int = 3,
) -> dict[str, Any]:
    """Measure Python pure-vector scorer latency without embedding time."""

    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != policy.dimension or vectors.shape[0] == 0:
        raise ValueError("embeddings must be a non-empty policy-aligned matrix")
    if repeats <= 0 or not quality_biases:
        raise ValueError("repeats and quality_biases must be non-empty")
    policy.route_vector(vectors[0], quality_bias=quality_biases[0])
    durations: list[float] = []
    for _ in range(repeats):
        for bias in quality_biases:
            for vector in vectors:
                started = time.perf_counter_ns()
                policy.route_vector(vector, quality_bias=bias)
                durations.append((time.perf_counter_ns() - started) / 1_000)
    values = np.asarray(durations, dtype=np.float64)
    return {
        "schema": "routerlab-scorer-latency-v1",
        "implementation": "python_numpy_pure_vector_scorer",
        "excludes_embedding": True,
        "routes": len(durations),
        "p50_microseconds": float(np.percentile(values, 50)),
        "p95_microseconds": float(np.percentile(values, 95)),
        "p99_microseconds": float(np.percentile(values, 99)),
        "mean_microseconds": float(np.mean(values)),
    }


def _sample_indices(prompt_keys: Sequence[str], sample_size: int, seed: int) -> NDArray[np.int64]:
    def rank(index: int) -> bytes:
        digest = hashlib.sha256()
        digest.update(b"routerlab-robustness-sample-v1\0")
        digest.update(seed.to_bytes(8, "big"))
        digest.update(prompt_keys[index].encode())
        return digest.digest()

    selected = sorted(range(len(prompt_keys)), key=rank)[:sample_size]
    return np.asarray(sorted(selected), dtype=np.int64)
