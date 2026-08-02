from __future__ import annotations

import hashlib

import numpy as np
import pytest

from routerlab.cli import build_parser, validate_run_provenance
from routerlab.schema import OutcomeMatrix, outcome_matrix_digest
from routerlab.split import Split, assign_prompt_splits


@pytest.mark.parametrize(
    "command",
    ["download", "prepare", "train", "evaluate", "diagnose", "report", "run-all"],
)
def test_cli_recognizes_every_documented_stage(command: str) -> None:
    parser = build_parser()

    arguments = parser.parse_args(
        [
            command,
            "--cache-root",
            ".cache-test",
            "--artifact-root",
            "artifacts-test",
            "--results-root",
            "results-test",
            "--seed",
            "7",
        ]
    )

    assert arguments.command == command
    assert arguments.seed == 7


def test_run_provenance_must_match_stage_seed_and_matrix() -> None:
    prompts = tuple(f"prompt {index}" for index in range(20))
    matrix = OutcomeMatrix(
        prompt_keys=tuple(hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts),
        datasets=tuple("demo" for _ in prompts),
        source_indices=tuple(str(index) for index in range(len(prompts))),
        prompts=prompts,
        models=("model-a",),
        quality=np.ones((len(prompts), 1), dtype=np.float32),
        realized_cost=np.ones((len(prompts), 1), dtype=np.float64),
        prompt_tokens=np.ones((len(prompts), 1), dtype=np.int64),
        completion_tokens=np.ones((len(prompts), 1), dtype=np.int64),
    )
    assignments = assign_prompt_splits(matrix.prompt_keys, 42)
    provenance = {
        "schema": "routerlab-provenance-v1",
        "archive_sha256": "b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1",
        "matrix_sha256": outcome_matrix_digest(matrix),
        "models": ["model-a"],
        "prompts": len(prompts),
        "split_seed": 42,
        "split_counts": {
            split.value: int(np.count_nonzero(assignments == split)) for split in Split
        },
    }

    validate_run_provenance(provenance, matrix, 42)
    with pytest.raises(ValueError, match="split seed"):
        validate_run_provenance(provenance, matrix, 43)
    with pytest.raises(ValueError, match="matrix SHA-256"):
        validate_run_provenance({**provenance, "matrix_sha256": "a" * 64}, matrix, 42)
