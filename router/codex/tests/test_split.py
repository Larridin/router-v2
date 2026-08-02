import hashlib

import numpy as np

from routerlab.split import Split, assign_prompt_splits, one_split_per_prompt


def keys_for(*prompts: str) -> tuple[str, ...]:
    return tuple(hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts)


def test_hash_split_is_deterministic_and_pinned() -> None:
    keys = keys_for("alpha", "beta", "gamma", "delta", "epsilon")

    first = assign_prompt_splits(keys, seed=42)
    second = assign_prompt_splits(keys, seed=42)

    assert first.tolist() == second.tolist()
    assert first.tolist() == [
        Split.TRAIN,
        Split.TRAIN,
        Split.TRAIN,
        Split.TEST,
        Split.TRAIN,
    ]


def test_duplicate_prompt_keys_never_cross_splits() -> None:
    alpha, beta = keys_for("alpha", "beta")
    keys = (alpha, beta, alpha, beta, alpha)

    splits = assign_prompt_splits(keys, seed=7)

    assert one_split_per_prompt(keys, splits)
    assert splits[0] == splits[2] == splits[4]
    assert splits[1] == splits[3]


def test_seed_changes_membership_and_large_sample_tracks_target_ratios() -> None:
    keys = keys_for(*(f"prompt-{index}" for index in range(10_000)))
    first = assign_prompt_splits(keys, seed=42)
    other = assign_prompt_splits(keys, seed=43)

    assert np.count_nonzero(first != other) > 1_000
    proportions = {split: np.mean(first == split) for split in Split}
    assert abs(proportions[Split.TRAIN] - 0.70) < 0.02
    assert abs(proportions[Split.VALIDATION] - 0.15) < 0.02
    assert abs(proportions[Split.TEST] - 0.15) < 0.02
