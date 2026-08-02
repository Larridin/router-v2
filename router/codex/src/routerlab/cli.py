"""Command-line workflow for the standalone routing experiment."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from routerlab.artifact import EmbeddingSpec, load_artifact, require_evaluation_context
from routerlab.data import (
    FLAGSHIP_MODELS,
    LLMROUTERBENCH_SHA256,
    LLMROUTERBENCH_URL,
    MATRIX_CACHE_SCHEMA,
    MATRIX_LOADER_REVISION,
    benchmark_selection_digest,
    discover_benchmark_files,
    download_archive,
    extract_archive,
    load_outcome_files,
    load_prepared_matrix_cache,
    save_prepared_matrix_cache,
)
from routerlab.diagnostics import measure_scorer_latency, robustness_diagnostics
from routerlab.embeddings import EmbeddingCache, FastEmbedEncoder
from routerlab.pipeline import evaluate_candidate, train_candidate
from routerlab.report import write_report
from routerlab.schema import OutcomeMatrix, outcome_matrix_digest
from routerlab.split import Split, assign_prompt_splits

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def build_parser() -> argparse.ArgumentParser:
    """Construct the documented staged command-line interface."""

    parser = argparse.ArgumentParser(
        prog="routerlab",
        description="Train and evaluate a standalone semantic-centroid model router.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "download",
        "prepare",
        "train",
        "evaluate",
        "diagnose",
        "report",
        "run-all",
    ):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--cache-root", type=Path, default=Path(".cache"))
        subparser.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
        subparser.add_argument("--results-root", type=Path, default=Path("results"))
        subparser.add_argument("--seed", type=int, default=42)
        subparser.add_argument(
            "--new-run-id",
            help="write frozen artifacts/results beneath this new run directory",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one CLI stage and return a process status."""

    arguments = build_parser().parse_args(argv)
    if arguments.seed < 0:
        raise SystemExit("--seed must be non-negative")
    artifact_root, results_root = _run_roots(
        arguments.artifact_root,
        arguments.results_root,
        arguments.new_run_id,
    )
    command = arguments.command
    if command == "download":
        result = download_archive(arguments.cache_root)
        _print(
            {
                "path": str(result.path),
                "reused": result.reused,
                "sha256": result.sha256,
                "size": result.size,
            }
        )
    elif command == "prepare":
        matrix, provenance = prepare(arguments.cache_root, results_root, arguments.seed)
        _print({"models": len(matrix.models), "prompts": matrix.n_prompts, **provenance["audit"]})
    elif command == "train":
        training = train(arguments.cache_root, artifact_root, results_root, arguments.seed)
        _print({"artifact_id": training.artifact_id, "path": str(training.artifact_directory)})
    elif command == "evaluate":
        evaluation = evaluate(arguments.cache_root, artifact_root, results_root, arguments.seed)
        _print({"artifact_id": evaluation.artifact_id, "path": str(evaluation.test_path)})
    elif command == "diagnose":
        paths = diagnose(arguments.cache_root, artifact_root, results_root, arguments.seed)
        _print({"robustness": str(paths[0]), "latency": str(paths[1])})
    elif command == "report":
        _print({"path": str(write_report(results_root))})
    elif command == "run-all":
        result = download_archive(arguments.cache_root)
        _print({"stage": "download", "reused": result.reused, "sha256": result.sha256})
        prepare(arguments.cache_root, results_root, arguments.seed)
        _print({"stage": "prepare"})
        training = train(arguments.cache_root, artifact_root, results_root, arguments.seed)
        _print({"stage": "train", "artifact_id": training.artifact_id})
        evaluate(arguments.cache_root, artifact_root, results_root, arguments.seed)
        _print({"stage": "evaluate"})
        diagnose(arguments.cache_root, artifact_root, results_root, arguments.seed)
        _print({"stage": "diagnose"})
        report = write_report(results_root)
        _print({"stage": "report", "path": str(report)})
    else:
        raise AssertionError(f"unhandled command: {command}")
    return 0


def prepare(
    cache_root: Path,
    results_root: Path,
    seed: int,
) -> tuple[OutcomeMatrix, dict[str, Any]]:
    """Extract, normalize, audit, and cache the public outcome matrix."""

    download = download_archive(cache_root)
    extracted = extract_archive(
        download.path,
        Path(cache_root) / "llmrouterbench" / "extracted",
    )
    files = discover_benchmark_files(extracted)
    matrix_path = _matrix_path(cache_root)
    manifest_path = _matrix_manifest_path(cache_root)
    selection_sha256 = benchmark_selection_digest(extracted, files)
    if matrix_path.exists() != manifest_path.exists():
        raise ValueError(
            "prepared matrix cache and manifest must either both exist or both be absent"
        )
    if matrix_path.exists():
        matrix, cache_manifest = load_prepared_matrix_cache(
            matrix_path,
            manifest_path,
            archive_sha256=download.sha256,
            selection_sha256=selection_sha256,
            expected_models=FLAGSHIP_MODELS,
        )
        cache_reused = True
    else:
        matrix, audit = load_outcome_files(files)
        cache_manifest = save_prepared_matrix_cache(
            matrix,
            matrix_path,
            manifest_path,
            archive_sha256=download.sha256,
            selection_sha256=selection_sha256,
            audit=audit,
        )
        cache_reused = False
    audit_dict = {
        **cache_manifest["audit"],
        "cache_reused": cache_reused,
        "selected_files": len(files),
    }
    assignments = assign_prompt_splits(matrix.prompt_keys, seed)
    provenance: dict[str, Any] = {
        "schema": "routerlab-provenance-v1",
        "dataset": "NPULH/LLMRouterBench",
        "dataset_url": LLMROUTERBENCH_URL,
        "archive_sha256": LLMROUTERBENCH_SHA256,
        "archive_size": download.size,
        "matrix_cache_schema": MATRIX_CACHE_SCHEMA,
        "matrix_loader_revision": MATRIX_LOADER_REVISION,
        "matrix_sha256": cache_manifest["matrix_sha256"],
        "selection_sha256": selection_sha256,
        "license": (
            "MIT benchmark repository; the Hugging Face archive has no separate "
            "dataset-card license declaration"
        ),
        "models": list(matrix.models),
        "datasets": sorted(set(matrix.datasets)),
        "prompts": matrix.n_prompts,
        "split_seed": seed,
        "split_counts": {
            split.value: int(np.count_nonzero(assignments == split)) for split in Split
        },
        "audit": audit_dict,
    }
    results = Path(results_root)
    results.mkdir(parents=True, exist_ok=True)
    provenance_path = results / "provenance.json"
    _write_json_new(provenance_path, provenance)
    return matrix, provenance


def train(cache_root: Path, artifact_root: Path, results_root: Path, seed: int):
    """Embed prompts, search validation configurations, and freeze a candidate."""

    matrix, vectors, encoder = _embedded_inputs(cache_root)
    provenance = _load_run_provenance(results_root, matrix, seed)
    return train_candidate(
        matrix,
        vectors,
        artifact_root=artifact_root,
        results_root=results_root,
        embedding=EmbeddingSpec(
            encoder.model_id, encoder.source_repo, encoder.revision, encoder.dimension
        ),
        seed=seed,
        trained_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        provenance={
            "archive_sha256": provenance["archive_sha256"],
            "dataset": provenance["dataset"],
            "matrix_loader_revision": provenance["matrix_loader_revision"],
            "selection_sha256": provenance["selection_sha256"],
        },
    )


def evaluate(cache_root: Path, artifact_root: Path, results_root: Path, seed: int):
    """Compare the frozen candidate with baselines on the held-out split."""

    matrix, vectors, encoder = _embedded_inputs(cache_root)
    _load_run_provenance(results_root, matrix, seed)
    return evaluate_candidate(
        matrix,
        vectors,
        artifact_directory=Path(artifact_root) / "codex-centroid-v1",
        results_root=results_root,
        embedding=EmbeddingSpec(
            encoder.model_id, encoder.source_repo, encoder.revision, encoder.dimension
        ),
        seed=seed,
    )


def diagnose(
    cache_root: Path,
    artifact_root: Path,
    results_root: Path,
    seed: int,
) -> tuple[Path, Path]:
    """Measure perturbation robustness and pure-vector scoring latency."""

    matrix, vectors, encoder = _embedded_inputs(cache_root)
    _load_run_provenance(results_root, matrix, seed)
    loaded = load_artifact(Path(artifact_root) / "codex-centroid-v1")
    assignments = assign_prompt_splits(matrix.prompt_keys, seed)
    require_evaluation_context(
        loaded.manifest,
        split_seed=seed,
        embedding=EmbeddingSpec(
            encoder.model_id, encoder.source_repo, encoder.revision, encoder.dimension
        ),
        training_matrix_sha256=outcome_matrix_digest(matrix.take(assignments != Split.TEST)),
    )
    test_mask = assignments == Split.TEST
    test = matrix.take(test_mask)
    robustness = robustness_diagnostics(
        test.prompts,
        test.prompt_keys,
        vectors[test_mask],
        test.quality,
        test.realized_cost,
        loaded.policy,
        encoder,
        seed=seed,
    )
    latency = measure_scorer_latency(
        loaded.policy,
        vectors[test_mask],
        quality_biases=(0.75, 1.0),
        repeats=3,
    )
    diagnostic_context = {
        "artifact_id": loaded.manifest.artifact_id,
        "matrix_sha256": outcome_matrix_digest(matrix),
        "seed": seed,
    }
    robustness.update(diagnostic_context)
    latency.update(diagnostic_context)
    results = Path(results_root)
    robustness_path = results / "robustness.json"
    latency_path = results / "latency.json"
    _write_json_new(robustness_path, robustness)
    _write_json_new(latency_path, latency)
    return robustness_path, latency_path


def _embedded_inputs(
    cache_root: Path,
) -> tuple[OutcomeMatrix, np.ndarray, FastEmbedEncoder]:
    matrix, _ = load_prepared_matrix_cache(
        _matrix_path(cache_root),
        _matrix_manifest_path(cache_root),
        archive_sha256=LLMROUTERBENCH_SHA256,
        expected_models=FLAGSHIP_MODELS,
    )
    root = Path(cache_root)
    encoder = FastEmbedEncoder(cache_dir=root / "fastembed-models")
    vectors = EmbeddingCache(root / "embeddings").get_or_encode(matrix.prompts, encoder)
    return matrix, vectors, encoder


def validate_run_provenance(
    provenance: dict[str, Any],
    matrix: OutcomeMatrix,
    seed: int,
) -> None:
    """Require every staged command to use prepare's exact matrix and split."""

    if provenance.get("schema") != "routerlab-provenance-v1":
        raise ValueError("run provenance schema does not match current workflow")
    if provenance.get("archive_sha256") != LLMROUTERBENCH_SHA256:
        raise ValueError("run provenance archive SHA-256 does not match pinned source")
    prepared_seed = provenance.get("split_seed")
    if not isinstance(prepared_seed, int) or isinstance(prepared_seed, bool):
        raise ValueError("run provenance has no valid split seed")
    if prepared_seed != seed:
        raise ValueError(
            f"stage split seed {seed} does not match prepared split seed {prepared_seed}"
        )
    if provenance.get("matrix_sha256") != outcome_matrix_digest(matrix):
        raise ValueError("run provenance matrix SHA-256 does not match prepared matrix")
    if provenance.get("models") != list(matrix.models):
        raise ValueError("run provenance model roster does not match prepared matrix")
    if provenance.get("prompts") != matrix.n_prompts:
        raise ValueError("run provenance prompt count does not match prepared matrix")
    assignments = assign_prompt_splits(matrix.prompt_keys, seed)
    expected_counts = {split.value: int(np.count_nonzero(assignments == split)) for split in Split}
    if provenance.get("split_counts") != expected_counts:
        raise ValueError("run provenance split counts do not match prepared matrix")


def _load_run_provenance(
    results_root: Path,
    matrix: OutcomeMatrix,
    seed: int,
) -> dict[str, Any]:
    provenance = _read_object(Path(results_root) / "provenance.json")
    validate_run_provenance(provenance, matrix, seed)
    return provenance


def _matrix_path(cache_root: Path) -> Path:
    return Path(cache_root) / "llmrouterbench" / "prepared" / "outcomes.npz"


def _matrix_manifest_path(cache_root: Path) -> Path:
    return Path(cache_root) / "llmrouterbench" / "prepared" / "outcomes.manifest.json"


def _run_roots(
    artifact_root: Path,
    results_root: Path,
    run_id: str | None,
) -> tuple[Path, Path]:
    if run_id is None:
        return artifact_root, results_root
    if _RUN_ID.fullmatch(run_id) is None:
        raise SystemExit("--new-run-id must contain only letters, digits, dot, underscore, or dash")
    return artifact_root / run_id, results_root / run_id


def _write_json_new(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen result: {path}")
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _print(value: object) -> None:
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
