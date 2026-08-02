"""Deterministic prompt-group train, validation, and test splits."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class Split(StrEnum):
    """One immutable experiment split."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def _bucket(prompt_digest: str, seed: int) -> int:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    try:
        prompt_bytes = bytes.fromhex(prompt_digest)
    except ValueError as error:
        raise ValueError("prompt key must be hexadecimal") from error
    if len(prompt_bytes) != hashlib.sha256().digest_size:
        raise ValueError("prompt key must be a SHA-256 digest")
    material = b"codex-router-split-v1\0" + seed.to_bytes(8, "big") + prompt_bytes
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], "big") % 10_000


def assign_prompt_splits(prompt_keys: Sequence[str], seed: int) -> NDArray[np.object_]:
    """Assign each prompt digest to a stable 70/15/15 split."""

    assignments: list[Split] = []
    for key in prompt_keys:
        bucket = _bucket(key, seed)
        if bucket < 7_000:
            assignments.append(Split.TRAIN)
        elif bucket < 8_500:
            assignments.append(Split.VALIDATION)
        else:
            assignments.append(Split.TEST)
    return np.asarray(assignments, dtype=object)


def one_split_per_prompt(prompt_keys: Iterable[str], splits: Iterable[Split]) -> bool:
    """Return whether every repeated prompt key has exactly one assignment."""

    seen: dict[str, Split] = {}
    for key, split in zip(prompt_keys, splits, strict=True):
        normalized = Split(split)
        previous = seen.setdefault(key, normalized)
        if previous != normalized:
            return False
    return True
