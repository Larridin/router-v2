from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from routerlab.artifact import EmbeddingSpec
from routerlab.data import prompt_key
from routerlab.pipeline import SearchSpace, evaluate_candidate, train_candidate
from routerlab.schema import OutcomeMatrix
from routerlab.split import Split, assign_prompt_splits


def synthetic_corpus(rows: int = 160) -> tuple[OutcomeMatrix, np.ndarray]:
    prompts = tuple(f"category {index % 2}: synthetic request {index}" for index in range(rows))
    keys = tuple(prompt_key(prompt) for prompt in prompts)
    quality = np.empty((rows, 2), dtype=np.float32)
    embeddings = np.empty((rows, 2), dtype=np.float32)
    for index in range(rows):
        category = index % 2
        quality[index] = [0.95, 0.05] if category == 0 else [0.05, 0.95]
        offset = 0.001 + index / (rows * 100)
        embeddings[index] = [1.0, offset] if category == 0 else [offset, 1.0]
    cost = np.full((rows, 2), 0.1, dtype=np.float64)
    tokens = np.full((rows, 2), 10, dtype=np.int64)
    matrix = OutcomeMatrix(
        prompt_keys=keys,
        datasets=tuple("synthetic" for _ in range(rows)),
        source_indices=tuple(str(index) for index in range(rows)),
        prompts=prompts,
        models=("model-a", "model-b"),
        quality=quality,
        realized_cost=cost,
        prompt_tokens=tokens,
        completion_tokens=tokens.copy(),
    )
    return matrix, embeddings


def search_space() -> SearchSpace:
    return SearchSpace(
        n_clusters=(2,),
        top_p=(1, 2),
        shrinkage=(1.0, 5.0),
        temperature=(0.05,),
        quality_biases=(0.25, 0.5, 0.75),
    )


def test_train_freezes_before_test_and_test_cannot_select_candidate(tmp_path: Path) -> None:
    matrix, embeddings = synthetic_corpus()
    split = assign_prompt_splits(matrix.prompt_keys, seed=42)
    hidden_changed = matrix.quality.copy()
    hidden_changed[split == Split.TEST] = hidden_changed[split == Split.TEST, ::-1]
    changed = OutcomeMatrix(
        prompt_keys=matrix.prompt_keys,
        datasets=matrix.datasets,
        source_indices=matrix.source_indices,
        prompts=matrix.prompts,
        models=matrix.models,
        quality=hidden_changed,
        realized_cost=matrix.realized_cost,
        prompt_tokens=matrix.prompt_tokens,
        completion_tokens=matrix.completion_tokens,
    )

    first = train_candidate(
        matrix,
        embeddings,
        artifact_root=tmp_path / "first-artifacts",
        results_root=tmp_path / "first-results",
        embedding=EmbeddingSpec("test/embedder", "test/source", "v1", 2),
        search_space=search_space(),
        seed=42,
        trained_at="2026-08-01T12:00:00Z",
    )
    second = train_candidate(
        changed,
        embeddings,
        artifact_root=tmp_path / "second-artifacts",
        results_root=tmp_path / "second-results",
        embedding=EmbeddingSpec("test/embedder", "test/source", "v1", 2),
        search_space=search_space(),
        seed=42,
        trained_at="2026-08-01T12:00:00Z",
    )

    assert first.events == ("search_completed", "artifact_frozen")
    assert not (tmp_path / "first-results" / "test_metrics.json").exists()
    assert first.artifact_id == second.artifact_id
    assert first.search_path.read_bytes() == second.search_path.read_bytes()
    assert (first.artifact_directory / "centroids.f32").read_bytes() == (
        second.artifact_directory / "centroids.f32"
    ).read_bytes()


def test_synthetic_end_to_end_beats_random_and_is_reproducible(tmp_path: Path) -> None:
    matrix, embeddings = synthetic_corpus()
    result_hashes: list[tuple[str, str]] = []

    for run in ("one", "two"):
        training = train_candidate(
            matrix,
            embeddings,
            artifact_root=tmp_path / run / "artifacts",
            results_root=tmp_path / run / "results",
            embedding=EmbeddingSpec("test/embedder", "test/source", "v1", 2),
            search_space=search_space(),
            seed=42,
            trained_at="2026-08-01T12:00:00Z",
        )
        evaluation = evaluate_candidate(
            matrix,
            embeddings,
            artifact_directory=training.artifact_directory,
            results_root=tmp_path / run / "results",
            embedding=EmbeddingSpec("test/embedder", "test/source", "v1", 2),
            seed=42,
        )
        candidate = next(
            point
            for point in evaluation.points
            if point["name"] == "codex_centroid" and point["quality_bias"] == 0.75
        )
        random = next(
            point
            for point in evaluation.points
            if point["name"] == "random" and point["quality_bias"] == 0.75
        )
        weave = next(
            point
            for point in evaluation.points
            if point["name"] == "weave_v075" and point["quality_bias"] == 0.75
        )

        assert evaluation.events == ("artifact_loaded", "test_evaluated")
        assert candidate["mean_quality"] > random["mean_quality"]
        assert candidate["macro_mean_quality"] == candidate["mean_quality"]
        assert candidate["macro_mean_cost"] == candidate["mean_cost"]
        payload = json.loads(evaluation.test_path.read_text())
        assert payload["schema"] == "routerlab-test-evaluation-v2"
        assert set(payload["bootstrap_by_bias"]) == {"0", "0.25", "0.5", "0.75", "1"}
        assert all(point["name"] != "oracle" for point in payload["frontier"])
        assert len(payload["oracle_points"]) == 5
        assert (
            payload["macro_candidate_cost_saving_at_best_single_quality"]
            == payload["candidate_cost_saving_at_best_single_quality"]
        )
        weave_baseline = payload["weave_baseline"]
        assert weave_baseline["config"] == {
            "n_clusters": 16,
            "top_p": 4,
            "shrinkage": 10.0,
            "seed": 42,
            "n_init": 10,
        }
        assert weave_baseline["trainer_source_revision"] == (
            "b74af8941cb71e3a0bc43af9d9a68b0c727fb7cf"
        )
        assert len(weave_baseline["policy_sha256"]) == 64
        assert set(weave_baseline["selection_sha256_by_bias"]) == {
            "0",
            "0.25",
            "0.5",
            "0.75",
            "1",
        }
        comparison = next(row for row in payload["weave_comparison"] if row["quality_bias"] == 0.75)
        assert (
            comparison["codex_quality_gain_vs_best_single"] == candidate["best_single_quality_gain"]
        )
        assert comparison["weave_quality_gain_vs_best_single"] == weave["best_single_quality_gain"]
        assert comparison["codex_minus_weave_quality"] == (
            candidate["mean_quality"] - weave["mean_quality"]
        )
        assert comparison["quality_bootstrap"]["replicates"] == 2000
        assert comparison["weave_quality_bootstrap_vs_best_single"]["replicates"] == 2000
        result_hashes.append(
            (
                training.artifact_id,
                hashlib.sha256(evaluation.test_path.read_bytes()).hexdigest(),
            )
        )

    assert result_hashes[0] == result_hashes[1]


def test_weave_fit_and_selections_do_not_depend_on_test_outcomes(tmp_path: Path) -> None:
    matrix, embeddings = synthetic_corpus()
    assignments = assign_prompt_splits(matrix.prompt_keys, seed=42)
    changed_quality = matrix.quality.copy()
    changed_cost = matrix.realized_cost.copy()
    changed_quality[assignments == Split.TEST] = 1.0 - changed_quality[assignments == Split.TEST]
    changed_cost[assignments == Split.TEST] *= 7.0
    changed = OutcomeMatrix(
        prompt_keys=matrix.prompt_keys,
        datasets=matrix.datasets,
        source_indices=matrix.source_indices,
        prompts=matrix.prompts,
        models=matrix.models,
        quality=changed_quality,
        realized_cost=changed_cost,
        prompt_tokens=matrix.prompt_tokens,
        completion_tokens=matrix.completion_tokens,
    )
    evaluations: list[dict[str, object]] = []
    for name, corpus in (("original", matrix), ("changed", changed)):
        training = train_candidate(
            corpus,
            embeddings,
            artifact_root=tmp_path / name / "artifacts",
            results_root=tmp_path / name / "results",
            embedding=EmbeddingSpec("test/embedder", "test/source", "v1", 2),
            search_space=search_space(),
            seed=42,
            trained_at="2026-08-01T12:00:00Z",
        )
        evaluation = evaluate_candidate(
            corpus,
            embeddings,
            artifact_directory=training.artifact_directory,
            results_root=training.search_path.parent,
            embedding=EmbeddingSpec("test/embedder", "test/source", "v1", 2),
            seed=42,
        )
        evaluations.append(json.loads(evaluation.test_path.read_text())["weave_baseline"])

    assert evaluations[0]["policy_sha256"] == evaluations[1]["policy_sha256"]
    assert evaluations[0]["selection_sha256_by_bias"] == evaluations[1]["selection_sha256_by_bias"]


def test_evaluation_rejects_a_different_split_seed(tmp_path: Path) -> None:
    matrix, embeddings = synthetic_corpus()
    training = train_candidate(
        matrix,
        embeddings,
        artifact_root=tmp_path / "artifacts",
        results_root=tmp_path / "train-results",
        embedding=EmbeddingSpec("test/embedder", "test/source", "v1", 2),
        search_space=search_space(),
        seed=42,
        trained_at="2026-08-01T12:00:00Z",
    )

    with pytest.raises(ValueError, match="split seed"):
        evaluate_candidate(
            matrix,
            embeddings,
            artifact_directory=training.artifact_directory,
            results_root=training.search_path.parent,
            embedding=EmbeddingSpec("test/embedder", "test/source", "v1", 2),
            seed=43,
        )


def test_evaluation_rejects_wrong_embedding_or_outcome_matrix(tmp_path: Path) -> None:
    matrix, embeddings = synthetic_corpus()
    training = train_candidate(
        matrix,
        embeddings,
        artifact_root=tmp_path / "artifacts",
        results_root=tmp_path / "train-results",
        embedding=EmbeddingSpec("test/embedder", "test/source", "v1", 2),
        search_space=search_space(),
        seed=42,
        trained_at="2026-08-01T12:00:00Z",
    )

    with pytest.raises(ValueError, match="embedding identity"):
        evaluate_candidate(
            matrix,
            embeddings,
            artifact_directory=training.artifact_directory,
            results_root=training.search_path.parent,
            embedding=EmbeddingSpec("wrong/embedder", "test/source", "v1", 2),
            seed=42,
        )

    changed_quality = matrix.quality.copy()
    assignments = assign_prompt_splits(matrix.prompt_keys, seed=42)
    changed_row = int(np.flatnonzero(assignments != Split.TEST)[0])
    changed_quality[changed_row, 0] = 0.0
    changed = OutcomeMatrix(
        prompt_keys=matrix.prompt_keys,
        datasets=matrix.datasets,
        source_indices=matrix.source_indices,
        prompts=matrix.prompts,
        models=matrix.models,
        quality=changed_quality,
        realized_cost=matrix.realized_cost,
        prompt_tokens=matrix.prompt_tokens,
        completion_tokens=matrix.completion_tokens,
    )
    with pytest.raises(ValueError, match="training matrix"):
        evaluate_candidate(
            changed,
            embeddings,
            artifact_directory=training.artifact_directory,
            results_root=training.search_path.parent,
            embedding=EmbeddingSpec("test/embedder", "test/source", "v1", 2),
            seed=42,
        )


def test_evaluation_rejects_search_record_for_another_artifact(tmp_path: Path) -> None:
    matrix, embeddings = synthetic_corpus()
    training = train_candidate(
        matrix,
        embeddings,
        artifact_root=tmp_path / "artifacts",
        results_root=tmp_path / "results",
        embedding=EmbeddingSpec("test/embedder", "test/source", "v1", 2),
        search_space=search_space(),
        seed=42,
        trained_at="2026-08-01T12:00:00Z",
    )
    search = json.loads(training.search_path.read_text())
    search["artifact_id"] = "a" * 64
    training.search_path.write_text(json.dumps(search), encoding="utf-8")

    with pytest.raises(ValueError, match="search artifact"):
        evaluate_candidate(
            matrix,
            embeddings,
            artifact_directory=training.artifact_directory,
            results_root=training.search_path.parent,
            embedding=EmbeddingSpec("test/embedder", "test/source", "v1", 2),
            seed=42,
        )
