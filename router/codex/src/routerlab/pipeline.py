"""Leakage-safe hyperparameter search, artifact freeze, and held-out evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from routerlab.artifact import (
    EmbeddingSpec,
    load_artifact,
    require_evaluation_context,
    write_artifact,
)
from routerlab.baselines import (
    GlobalUtilityPolicy,
    RandomPolicy,
    VectorPolicy,
    constant_baselines,
    fit_knn_policy,
    fit_ridge_policy,
    oracle_select,
)
from routerlab.centroid import (
    CentroidConfig,
    CentroidPolicy,
    build_centroid_policy,
    fit_centroid_geometry,
)
from routerlab.metrics import (
    cost_saving_at_quality,
    evaluate_selection,
    normalized_frontier_area,
    paired_bootstrap_difference,
    pareto_frontier,
    selected_outcomes,
)
from routerlab.schema import OutcomeMatrix, outcome_matrix_digest
from routerlab.split import Split, assign_prompt_splits, one_split_per_prompt
from routerlab.weave import WeaveConfig, fit_weave_policy

DEFAULT_SEARCH_SPACE: SearchSpace


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """Finite, declared centroid hyperparameter search space."""

    n_clusters: tuple[int, ...] = (8, 16, 32, 64)
    top_p: tuple[int, ...] = (1, 2, 4)
    shrinkage: tuple[float, ...] = (1.0, 5.0, 10.0, 25.0)
    temperature: tuple[float, ...] = (0.02, 0.05, 0.1)
    quality_biases: tuple[float, ...] = (0.25, 0.5, 0.75)

    def __post_init__(self) -> None:
        fields = (
            self.n_clusters,
            self.top_p,
            self.shrinkage,
            self.temperature,
            self.quality_biases,
        )
        if any(not values or len(set(values)) != len(values) for values in fields):
            raise ValueError("search dimensions must be non-empty and unique")
        if any(value <= 0 for value in self.n_clusters + self.top_p):
            raise ValueError("cluster counts and top_p must be positive")
        if any(not math.isfinite(value) or value <= 0 for value in self.shrinkage):
            raise ValueError("shrinkage values must be positive and finite")
        if any(not math.isfinite(value) or value <= 0 for value in self.temperature):
            raise ValueError("temperatures must be positive and finite")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in self.quality_biases):
            raise ValueError("quality biases must be finite and in [0, 1]")
        object.__setattr__(self, "n_clusters", tuple(sorted(self.n_clusters)))
        object.__setattr__(self, "top_p", tuple(sorted(self.top_p)))
        object.__setattr__(self, "shrinkage", tuple(sorted(self.shrinkage)))
        object.__setattr__(self, "temperature", tuple(sorted(self.temperature)))
        object.__setattr__(self, "quality_biases", tuple(sorted(self.quality_biases)))


DEFAULT_SEARCH_SPACE = SearchSpace()


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """Paths and state transitions from a completed search and freeze."""

    artifact_id: str
    artifact_directory: Path
    search_path: Path
    events: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    """Paths and aggregate points from one held-out evaluation."""

    artifact_id: str
    test_path: Path
    points: tuple[dict[str, Any], ...]
    events: tuple[str, ...]


def train_candidate(
    matrix: OutcomeMatrix,
    embeddings: NDArray[np.floating],
    *,
    artifact_root: Path,
    results_root: Path,
    embedding: EmbeddingSpec,
    search_space: SearchSpace = DEFAULT_SEARCH_SPACE,
    seed: int = 42,
    trained_at: str,
    provenance: dict[str, Any] | None = None,
) -> TrainingRun:
    """Search on validation outcomes and freeze a train-fitted candidate."""

    vectors = _validate_inputs(matrix, embeddings, embedding.dimension)
    assignments = assign_prompt_splits(matrix.prompt_keys, seed)
    _validate_splits(matrix, assignments)
    train_mask = assignments == Split.TRAIN
    validation_mask = assignments == Split.VALIDATION
    train = matrix.take(train_mask)
    validation = matrix.take(validation_mask)
    train_vectors = vectors[train_mask]
    validation_vectors = vectors[validation_mask]
    best_single_index = int(np.argmax(np.mean(train.quality, axis=0)))

    records: list[dict[str, Any]] = []
    policies: dict[tuple[int, int, float, float], CentroidPolicy] = {}
    for n_clusters in search_space.n_clusters:
        if n_clusters > train.n_prompts:
            continue
        geometry = fit_centroid_geometry(
            train_vectors,
            train.quality,
            train.realized_cost,
            train.models,
            n_clusters=n_clusters,
            seed=seed,
        )
        for top_p in search_space.top_p:
            if top_p > n_clusters:
                continue
            for shrinkage in search_space.shrinkage:
                for temperature in search_space.temperature:
                    key = (n_clusters, top_p, shrinkage, temperature)
                    policy = build_centroid_policy(
                        geometry,
                        top_p=top_p,
                        shrinkage=shrinkage,
                        temperature=temperature,
                    )
                    policies[key] = policy
                    validation_metrics: list[dict[str, Any]] = []
                    regrets: list[float] = []
                    for bias in search_space.quality_biases:
                        selections = policy.select_many(validation_vectors, bias)
                        metrics = evaluate_selection(
                            "codex_centroid",
                            validation.quality,
                            validation.realized_cost,
                            selections,
                            best_single_index=best_single_index,
                            quality_bias=bias,
                        )
                        metrics_dict = asdict(metrics)
                        validation_metrics.append(metrics_dict)
                        regrets.append(metrics.mean_normalized_oracle_regret)
                    records.append(
                        {
                            "config": _config_dict(policy.config),
                            "objective": float(np.mean(regrets)),
                            "validation": validation_metrics,
                        }
                    )
    if not records:
        raise ValueError("search space has no configurations valid for the training rows")
    winner_record = min(
        records,
        key=lambda record: (
            record["objective"],
            record["config"]["n_clusters"],
            record["config"]["top_p"],
            record["config"]["shrinkage"],
            record["config"]["temperature"],
        ),
    )
    winner_config = winner_record["config"]
    winner_key = (
        winner_config["n_clusters"],
        winner_config["top_p"],
        winner_config["shrinkage"],
        winner_config["temperature"],
    )
    winner = policies[winner_key]

    results_directory = Path(results_root)
    results_directory.mkdir(parents=True, exist_ok=True)
    search_path = results_directory / "search.json"
    if search_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen result: {search_path}")
    search_payload = {
        "schema": "routerlab-search-v2",
        "seed": seed,
        "models": list(matrix.models),
        "split_counts": _split_counts(assignments),
        "selection_metric": "mean validation normalized oracle regret",
        "selected": winner_record,
        "configurations": records,
    }

    artifact_directory = Path(artifact_root) / "codex-centroid-v1"
    artifact_provenance = dict(provenance or {})
    artifact_provenance.pop("matrix_sha256", None)
    artifact_provenance.update(
        {
            "training_matrix_sha256": outcome_matrix_digest(
                matrix.take(train_mask | validation_mask)
            ),
            "split_seed": seed,
            "train_rows": train.n_prompts,
            "validation_rows": validation.n_prompts,
            "selection_objective": format(winner_record["objective"], ".17g"),
        }
    )
    manifest = write_artifact(
        winner,
        artifact_directory,
        embedding=embedding,
        trained_at=trained_at,
        provenance=artifact_provenance,
    )
    search_payload["artifact_id"] = manifest.artifact_id
    _write_json_new(search_path, search_payload)
    return TrainingRun(
        artifact_id=manifest.artifact_id,
        artifact_directory=artifact_directory,
        search_path=search_path,
        events=("search_completed", "artifact_frozen"),
    )


def evaluate_candidate(
    matrix: OutcomeMatrix,
    embeddings: NDArray[np.floating],
    *,
    artifact_directory: Path,
    results_root: Path,
    embedding: EmbeddingSpec,
    seed: int = 42,
    quality_biases: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> EvaluationRun:
    """Evaluate a frozen artifact and train-only baselines on held-out rows."""

    loaded = load_artifact(artifact_directory)
    if loaded.manifest.models != matrix.models:
        raise ValueError("artifact model roster does not match outcome matrix")
    search = _read_json_object(Path(results_root) / "search.json")
    if search.get("schema") != "routerlab-search-v2":
        raise ValueError("search result schema does not match current workflow")
    if search.get("artifact_id") != loaded.manifest.artifact_id:
        raise ValueError("search artifact does not match evaluated artifact")
    selected = search.get("selected")
    if not isinstance(selected, dict) or selected.get("config") != _config_dict(
        loaded.manifest.config
    ):
        raise ValueError("search winner config does not match evaluated artifact")
    assignments = assign_prompt_splits(matrix.prompt_keys, seed)
    _validate_splits(matrix, assignments)
    training_matrix_sha256 = outcome_matrix_digest(matrix.take(assignments != Split.TEST))
    require_evaluation_context(
        loaded.manifest,
        split_seed=seed,
        embedding=embedding,
        training_matrix_sha256=training_matrix_sha256,
    )
    vectors = _validate_inputs(matrix, embeddings, loaded.manifest.embedding.dimension)
    train_mask = assignments == Split.TRAIN
    test_mask = assignments == Split.TEST
    train = matrix.take(train_mask)
    test = matrix.take(test_mask)
    train_vectors = vectors[train_mask]
    test_vectors = vectors[test_mask]

    cheapest, best_single = constant_baselines(train.quality, train.realized_cost)
    best_single_index = best_single.model_index
    weave = fit_weave_policy(
        train_vectors,
        train.quality,
        train.realized_cost,
        train.models,
        config=WeaveConfig.v075(),
    )
    policies: tuple[VectorPolicy, ...] = (
        RandomPolicy(matrix.n_models, seed),
        cheapest,
        best_single,
        GlobalUtilityPolicy.fit(train.quality, train.realized_cost),
        weave,
        fit_ridge_policy(train_vectors, train.quality, train.realized_cost),
        fit_knn_policy(
            train_vectors,
            train.quality,
            train.realized_cost,
            neighbors=min(5, train.n_prompts),
        ),
    )
    points: list[dict[str, Any]] = []
    outcomes: dict[tuple[str, float], tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    selections_by_policy: dict[tuple[str, float], NDArray[np.int64]] = {}
    for name, selector in (("codex_centroid", loaded.policy), *[(p.name, p) for p in policies]):
        for bias in quality_biases:
            if isinstance(selector, CentroidPolicy):
                selections = selector.select_many(test_vectors, bias)
            else:
                selections = selector.select(test_vectors, bias)
            metrics = evaluate_selection(
                name,
                test.quality,
                test.realized_cost,
                selections,
                best_single_index=best_single_index,
                quality_bias=bias,
            )
            point = asdict(metrics)
            point.update(_macro_outcomes(test, selections))
            points.append(point)
            outcomes[(name, bias)] = selected_outcomes(test.quality, test.realized_cost, selections)
            selections_by_policy[(name, bias)] = selections
    for bias in quality_biases:
        selections = oracle_select(test.quality, test.realized_cost, quality_bias=bias)
        metrics = evaluate_selection(
            "oracle",
            test.quality,
            test.realized_cost,
            selections,
            best_single_index=best_single_index,
            quality_bias=bias,
        )
        point = asdict(metrics)
        point.update(_macro_outcomes(test, selections))
        points.append(point)
        outcomes[("oracle", bias)] = selected_outcomes(test.quality, test.realized_cost, selections)

    servable_indices = [index for index, point in enumerate(points) if point["name"] != "oracle"]
    servable_quality = np.asarray([points[index]["mean_quality"] for index in servable_indices])
    servable_cost = np.asarray([points[index]["mean_cost"] for index in servable_indices])
    frontier_indices = [
        servable_indices[int(index)] for index in pareto_frontier(servable_quality, servable_cost)
    ]
    reference = next(
        point for point in points if point["name"] == "best_single" and point["quality_bias"] == 0.5
    )
    candidate_points = [point for point in points if point["name"] == "codex_centroid"]
    saving = cost_saving_at_quality(
        quality=np.asarray([point["mean_quality"] for point in candidate_points]),
        cost=np.asarray([point["mean_cost"] for point in candidate_points]),
        reference_quality=reference["mean_quality"],
        reference_cost=reference["mean_cost"],
    )
    macro_saving = cost_saving_at_quality(
        quality=np.asarray([point["macro_mean_quality"] for point in candidate_points]),
        cost=np.asarray([point["macro_mean_cost"] for point in candidate_points]),
        reference_quality=reference["macro_mean_quality"],
        reference_cost=reference["macro_mean_cost"],
    )
    weave_points = [point for point in points if point["name"] == weave.name]
    weave_saving = cost_saving_at_quality(
        quality=np.asarray([point["mean_quality"] for point in weave_points]),
        cost=np.asarray([point["mean_cost"] for point in weave_points]),
        reference_quality=reference["mean_quality"],
        reference_cost=reference["mean_cost"],
    )
    macro_weave_saving = cost_saving_at_quality(
        quality=np.asarray([point["macro_mean_quality"] for point in weave_points]),
        cost=np.asarray([point["macro_mean_cost"] for point in weave_points]),
        reference_quality=reference["macro_mean_quality"],
        reference_cost=reference["macro_mean_cost"],
    )
    comparison_bias = min(quality_biases, key=lambda value: abs(value - 0.75))
    candidate_quality, candidate_cost = outcomes[("codex_centroid", comparison_bias)]
    baseline_quality, baseline_cost = outcomes[("best_single", comparison_bias)]
    bootstrap = {
        "quality_vs_best_single": asdict(
            paired_bootstrap_difference(candidate_quality, baseline_quality, seed=seed)
        ),
        "cost_vs_best_single": asdict(
            paired_bootstrap_difference(candidate_cost, baseline_cost, seed=seed)
        ),
    }
    bootstrap_by_bias: dict[str, dict[str, Any]] = {}
    weave_comparison: list[dict[str, Any]] = []
    for bias in quality_biases:
        bias_candidate_quality, bias_candidate_cost = outcomes[("codex_centroid", bias)]
        bias_baseline_quality, bias_baseline_cost = outcomes[("best_single", bias)]
        bootstrap_by_bias[f"{bias:g}"] = {
            "quality_vs_best_single": asdict(
                paired_bootstrap_difference(
                    bias_candidate_quality, bias_baseline_quality, seed=seed
                )
            ),
            "cost_vs_best_single": asdict(
                paired_bootstrap_difference(bias_candidate_cost, bias_baseline_cost, seed=seed)
            ),
        }
        weave_quality, weave_cost = outcomes[(weave.name, bias)]
        codex_point = next(point for point in candidate_points if point["quality_bias"] == bias)
        weave_point = next(point for point in weave_points if point["quality_bias"] == bias)
        weave_comparison.append(
            {
                "quality_bias": bias,
                "codex_mean_quality": codex_point["mean_quality"],
                "weave_mean_quality": weave_point["mean_quality"],
                "codex_mean_cost": codex_point["mean_cost"],
                "weave_mean_cost": weave_point["mean_cost"],
                "codex_quality_gain_vs_best_single": codex_point["best_single_quality_gain"],
                "weave_quality_gain_vs_best_single": weave_point["best_single_quality_gain"],
                "codex_minus_weave_quality": (
                    codex_point["mean_quality"] - weave_point["mean_quality"]
                ),
                "codex_minus_weave_cost": (codex_point["mean_cost"] - weave_point["mean_cost"]),
                "codex_quality_bootstrap_vs_best_single": bootstrap_by_bias[f"{bias:g}"][
                    "quality_vs_best_single"
                ],
                "weave_quality_bootstrap_vs_best_single": asdict(
                    paired_bootstrap_difference(
                        weave_quality,
                        bias_baseline_quality,
                        seed=seed,
                    )
                ),
                "quality_bootstrap": asdict(
                    paired_bootstrap_difference(
                        bias_candidate_quality,
                        weave_quality,
                        seed=seed,
                    )
                ),
                "cost_bootstrap": asdict(
                    paired_bootstrap_difference(
                        bias_candidate_cost,
                        weave_cost,
                        seed=seed,
                    )
                ),
            }
        )
    per_dataset = _per_dataset_candidate(
        test,
        test_vectors,
        loaded.policy,
        quality_biases,
        best_single_index,
    )
    payload = {
        "schema": "routerlab-test-evaluation-v2",
        "artifact_id": loaded.manifest.artifact_id,
        "embedding": asdict(embedding),
        "matrix_sha256": outcome_matrix_digest(matrix),
        "training_matrix_sha256": training_matrix_sha256,
        "seed": seed,
        "test_rows": test.n_prompts,
        "aggregation": {
            "mean_quality": "prompt-weighted micro average",
            "macro_mean_quality": "equal-weight average of dataset means",
            "mean_cost": "prompt-weighted micro average USD per request",
            "macro_mean_cost": "equal-weight average of dataset mean USD per request",
        },
        "points": points,
        "frontier": [points[index] for index in frontier_indices],
        "oracle_points": [point for point in points if point["name"] == "oracle"],
        "normalized_frontier_area": normalized_frontier_area(servable_quality, servable_cost),
        "candidate_cost_saving_at_best_single_quality": saving,
        "macro_candidate_cost_saving_at_best_single_quality": macro_saving,
        "weave_cost_saving_at_best_single_quality": weave_saving,
        "macro_weave_cost_saving_at_best_single_quality": macro_weave_saving,
        "weave_baseline": {
            "name": weave.name,
            "algorithm": "Weave v0.75 core cluster recipe retrained on shared public data",
            "source": "internal/router/cluster/artifacts/v0.75/metadata.yaml",
            "trainer_source_revision": "b74af8941cb71e3a0bc43af9d9a68b0c727fb7cf",
            "trainer_sources": [
                "scripts/train_cluster_router.py",
                "scripts/bench_walker.py",
            ],
            "config": asdict(weave.config),
            "policy_sha256": weave.digest,
            "selection_sha256_by_bias": {
                f"{bias:g}": _selection_digest(selections_by_policy[(weave.name, bias)])
                for bias in quality_biases
            },
        },
        "weave_comparison": weave_comparison,
        "bootstrap_bias": comparison_bias,
        "bootstrap": bootstrap,
        "bootstrap_by_bias": bootstrap_by_bias,
        "per_dataset": per_dataset,
    }
    results_directory = Path(results_root)
    results_directory.mkdir(parents=True, exist_ok=True)
    test_path = results_directory / "test_metrics.json"
    _write_json_new(test_path, payload)
    return EvaluationRun(
        artifact_id=loaded.manifest.artifact_id,
        test_path=test_path,
        points=tuple(points),
        events=("artifact_loaded", "test_evaluated"),
    )


def _per_dataset_candidate(
    test: OutcomeMatrix,
    embeddings: NDArray[np.float32],
    policy: CentroidPolicy,
    quality_biases: tuple[float, ...],
    best_single_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    datasets = sorted(set(test.datasets))
    dataset_array = np.asarray(test.datasets)
    for dataset in datasets:
        mask = dataset_array == dataset
        subset = test.take(mask)
        for bias in quality_biases:
            selections = policy.select_many(embeddings[mask], bias)
            metrics = evaluate_selection(
                "codex_centroid",
                subset.quality,
                subset.realized_cost,
                selections,
                best_single_index=best_single_index,
                quality_bias=bias,
            )
            rows.append({"dataset": dataset, **asdict(metrics)})
    return rows


def _macro_outcomes(
    matrix: OutcomeMatrix,
    selections: NDArray[np.integer],
) -> dict[str, float]:
    selected_quality, selected_cost = selected_outcomes(
        matrix.quality, matrix.realized_cost, selections
    )
    datasets = np.asarray(matrix.datasets)
    quality_means: list[float] = []
    cost_means: list[float] = []
    for dataset in sorted(set(matrix.datasets)):
        mask = datasets == dataset
        quality_means.append(float(np.mean(selected_quality[mask])))
        cost_means.append(float(np.mean(selected_cost[mask])))
    return {
        "macro_mean_quality": float(np.mean(quality_means)),
        "macro_mean_cost": float(np.mean(cost_means)),
    }


def _selection_digest(selections: NDArray[np.integer]) -> str:
    values = np.ascontiguousarray(selections, dtype="<i8")
    digest = hashlib.sha256(b"routerlab-model-selections-v1\0")
    digest.update(len(values).to_bytes(8, "big"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _validate_inputs(
    matrix: OutcomeMatrix,
    embeddings: NDArray[np.floating],
    dimension: int,
) -> NDArray[np.float32]:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.shape != (matrix.n_prompts, dimension):
        raise ValueError("embeddings must align with outcomes and artifact dimension")
    if not np.all(np.isfinite(values)):
        raise ValueError("embeddings must be finite")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("embeddings contain a zero row")
    return np.asarray(values / norms, dtype=np.float32)


def _validate_splits(matrix: OutcomeMatrix, assignments: NDArray[np.object_]) -> None:
    if not one_split_per_prompt(matrix.prompt_keys, assignments):
        raise ValueError("a prompt appears in more than one split")
    missing = [split.value for split in Split if not np.any(assignments == split)]
    if missing:
        raise ValueError(f"empty experiment splits: {missing}")


def _split_counts(assignments: NDArray[np.object_]) -> dict[str, int]:
    return {split.value: int(np.count_nonzero(assignments == split)) for split in Split}


def _config_dict(config: CentroidConfig) -> dict[str, int | float]:
    return {
        "n_clusters": config.n_clusters,
        "top_p": config.top_p,
        "shrinkage": config.shrinkage,
        "temperature": config.temperature,
        "seed": config.seed,
    }


def _write_json_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen result: {path}")
    encoded = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    path.write_text(encoded + "\n", encoding="utf-8")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read frozen result {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"frozen result {path} must contain an object")
    return value
