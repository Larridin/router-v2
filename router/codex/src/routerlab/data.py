"""Normalize public benchmark records into a full-information matrix."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tarfile
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
import numpy as np

from routerlab.schema import LoadAudit, OutcomeMatrix, outcome_matrix_digest

LLMROUTERBENCH_URL = (
    "https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/main/"
    "bench-release.tar.gz?download=true"
)
LLMROUTERBENCH_SHA256 = "b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1"
FLAGSHIP_DATASETS = (
    "aime",
    "arenahard",
    "gpqa",
    "hle",
    "livecodebench",
    "livemathbench",
    "mmlupro",
    "simpleqa",
    "swe-bench",
    "tau2",
)
FLAGSHIP_MODELS = (
    "claude-sonnet-4",
    "deepseek-r1-0528",
    "deepseek-v3-0324",
    "deepseek-v3.1-terminus",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "glm-4.6",
    "gpt-5",
    "gpt-5-chat",
    "intern-s1",
    "kimi-k2-0905",
    "qwen3-235b-a22b-2507",
    "qwen3-235b-a22b-thinking-2507",
)
FLAGSHIP_SPLITS = ("hybrid", "test", "test_3000", "v1", "verified")
MATRIX_CACHE_SCHEMA = "routerlab-matrix-cache-v1"
MATRIX_LOADER_REVISION = "llmrouterbench-flagship-v1"
MATRIX_FILE_SCHEMA = "routerlab-outcome-matrix-file-v2"


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Verified local benchmark archive metadata."""

    path: Path
    sha256: str
    size: int
    reused: bool


@dataclass(frozen=True, slots=True)
class _Outcome:
    dataset: str
    source_index: str
    prompt: str
    model: str
    quality: float
    cost: float
    prompt_tokens: int
    completion_tokens: int


def normalize_prompt(prompt: str) -> str:
    """Normalize line endings and outer whitespace without changing content."""

    return prompt.replace("\r\n", "\n").replace("\r", "\n").strip()


def prompt_key(prompt: str) -> str:
    """Return the stable SHA-256 identity of normalized prompt text."""

    return hashlib.sha256(normalize_prompt(prompt).encode("utf-8")).hexdigest()


def _finite_float(record: dict[str, Any], field: str) -> float:
    try:
        value = float(record[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid {field}") from error
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field}")
    return value


def _token_count(record: dict[str, Any], field: str, normalizations: Counter[str]) -> int:
    try:
        value = int(record[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid {field}") from error
    if value < 0:
        normalizations[f"negative_{field}_to_zero"] += 1
        return 0
    return value


def _parse_record(record: dict[str, Any], normalizations: Counter[str]) -> _Outcome:
    try:
        dataset = str(record["dataset"]).strip()
        model = str(record["model"]).strip()
        source_index = str(record["index"])
        prompt = normalize_prompt(_prompt_string(record["prompt"]))
    except KeyError as error:
        raise ValueError(f"missing {error.args[0]}") from error
    if not dataset or not model or not prompt:
        raise ValueError("dataset, model, and prompt must be non-empty")
    cost = _finite_float(record, "cost")
    if cost < 0:
        raise ValueError("negative cost")
    return _Outcome(
        dataset=dataset,
        source_index=source_index,
        prompt=prompt,
        model=model,
        quality=_finite_float(record, "score"),
        cost=cost,
        prompt_tokens=_token_count(record, "prompt_tokens", normalizations),
        completion_tokens=_token_count(record, "completion_tokens", normalizations),
    )


def _records_from_file(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a records list")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{path}: every record must be an object")
    if not isinstance(payload, dict):
        return records
    dataset = payload.get("dataset_name")
    model = payload.get("model_name")
    if dataset is None and model is None:
        return records
    if not isinstance(dataset, str) or not dataset or not isinstance(model, str) or not model:
        raise ValueError(f"{path}: invalid benchmark file metadata")
    inherited: list[dict[str, Any]] = []
    for record in records:
        normalized = dict(record)
        normalized.setdefault("dataset", dataset)
        normalized.setdefault("model", model)
        inherited.append(normalized)
    return inherited


def load_outcome_files(paths: Iterable[Path]) -> tuple[OutcomeMatrix, LoadAudit]:
    """Load complete prompt rows shared by the union model roster."""

    ordered_paths = tuple(sorted((Path(path) for path in paths), key=lambda path: str(path)))
    if not ordered_paths:
        raise ValueError("at least one outcome file is required")

    raw_records = 0
    outcomes: list[_Outcome] = []
    normalizations: Counter[str] = Counter()
    for path in ordered_paths:
        records = _records_from_file(path)
        raw_records += len(records)
        for record in records:
            try:
                outcomes.append(_parse_record(record, normalizations))
            except ValueError as error:
                raise ValueError(f"{path}: {error}") from error

    models = tuple(sorted({outcome.model for outcome in outcomes}))
    if not models:
        raise ValueError("outcome files contain no models")
    model_index = {model: index for index, model in enumerate(models)}

    grouped: dict[str, dict[str, _Outcome]] = {}
    order: list[str] = []
    rejected: Counter[str] = Counter()
    invalid_keys: set[str] = set()
    for outcome in outcomes:
        key = prompt_key(outcome.prompt)
        if key not in grouped:
            grouped[key] = {}
            order.append(key)
        existing = grouped[key].get(outcome.model)
        if existing is not None and existing != outcome:
            invalid_keys.add(key)
            continue
        grouped[key][outcome.model] = outcome

    accepted_keys: list[str] = []
    for key in order:
        if key in invalid_keys:
            rejected["conflicting_duplicate"] += 1
        elif set(grouped[key]) != set(models):
            rejected["incomplete_roster"] += 1
        else:
            accepted_keys.append(key)
    if not accepted_keys:
        raise ValueError("no complete prompt rows remain")

    rows = len(accepted_keys)
    columns = len(models)
    quality = np.empty((rows, columns), dtype=np.float32)
    cost = np.empty((rows, columns), dtype=np.float64)
    prompt_tokens = np.empty((rows, columns), dtype=np.int64)
    completion_tokens = np.empty((rows, columns), dtype=np.int64)
    datasets: list[str] = []
    source_indices: list[str] = []
    prompts: list[str] = []

    for row, key in enumerate(accepted_keys):
        by_model = grouped[key]
        representative = by_model[models[0]]
        datasets.append(representative.dataset)
        source_indices.append(representative.source_index)
        prompts.append(representative.prompt)
        for model, outcome in by_model.items():
            column = model_index[model]
            quality[row, column] = outcome.quality
            cost[row, column] = outcome.cost
            prompt_tokens[row, column] = outcome.prompt_tokens
            completion_tokens[row, column] = outcome.completion_tokens

    matrix = OutcomeMatrix(
        prompt_keys=tuple(accepted_keys),
        datasets=tuple(datasets),
        source_indices=tuple(source_indices),
        prompts=tuple(prompts),
        models=models,
        quality=quality,
        realized_cost=cost,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    audit = LoadAudit(
        files=len(ordered_paths),
        raw_records=raw_records,
        accepted_prompts=matrix.n_prompts,
        rejections=dict(sorted(rejected.items())),
        normalizations=dict(sorted(normalizations.items())),
    )
    return matrix, audit


def download_archive(
    cache_root: Path,
    *,
    client: httpx.Client | None = None,
    url: str = LLMROUTERBENCH_URL,
    expected_sha256: str = LLMROUTERBENCH_SHA256,
) -> DownloadResult:
    """Download the pinned public archive and verify its SHA-256 digest."""

    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    directory = Path(cache_root) / "llmrouterbench"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "bench-release.tar.gz"
    if archive.exists():
        digest, size = _file_digest(archive)
        if digest == expected_sha256:
            return DownloadResult(archive, digest, size, True)

    temporary = directory / "bench-release.tar.gz.part"
    owns_client = client is None
    active_client = client or httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0),
    )
    digest = hashlib.sha256()
    size = 0
    try:
        with active_client.stream("GET", url) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
    finally:
        if owns_client:
            active_client.close()
    actual = digest.hexdigest()
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"archive SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    os.replace(temporary, archive)
    return DownloadResult(archive, actual, size, False)


def extract_archive(archive: Path, destination: Path) -> Path:
    """Safely extract a verified tar archive without links or path traversal."""

    source = Path(archive)
    digest, _ = _file_digest(source)
    output = Path(destination)
    marker = output / ".routerlab-extraction.json"
    if marker.exists():
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid extraction marker: {error}") from error
        if existing == {"archive_sha256": digest, "schema": "routerlab-extraction-v1"}:
            return output
        raise ValueError("extraction marker does not match archive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to extract into non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(source, mode="r:gz") as handle:
            members = handle.getmembers()
            for member in members:
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or member.issym()
                    or member.islnk()
                    or not (member.isdir() or member.isfile())
                ):
                    raise ValueError(f"unsafe archive member: {member.name}")
            handle.extractall(output, members=members, filter="data")
    except (OSError, tarfile.TarError) as error:
        raise ValueError(f"cannot extract archive: {error}") from error
    marker.write_text(
        json.dumps(
            {"archive_sha256": digest, "schema": "routerlab-extraction-v1"},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def discover_benchmark_files(
    root: Path,
    *,
    datasets: tuple[str, ...] = FLAGSHIP_DATASETS,
    models: tuple[str, ...] = FLAGSHIP_MODELS,
    splits: tuple[str, ...] = FLAGSHIP_SPLITS,
) -> tuple[Path, ...]:
    """Find the newest benchmark file for each requested dataset/model/split."""

    dataset_set = set(datasets)
    model_set = set(models)
    split_set = set(splits)
    grouped: dict[tuple[str, str, str], list[Path]] = {}
    for path in sorted(Path(root).rglob("*.json")):
        if path.name == ".routerlab-extraction.json":
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
            continue
        dataset = payload.get("dataset_name")
        model = payload.get("model_name")
        split = payload.get("split")
        if (
            dataset in dataset_set
            and model in model_set
            and split in split_set
            and not payload.get("demo", False)
        ):
            grouped.setdefault((dataset, split, model), []).append(path)
    selected = [_latest_file(paths) for paths in grouped.values()]
    if not selected:
        raise ValueError("no requested benchmark result files found")
    found_models = {_benchmark_identity(path)[2] for path in selected}
    missing_models = model_set - found_models
    if missing_models:
        raise ValueError(f"benchmark archive is missing requested models: {sorted(missing_models)}")
    return tuple(sorted(selected, key=str))


def save_outcome_matrix(matrix: OutcomeMatrix, path: Path) -> None:
    """Save an outcome matrix to a compressed, pickle-free NumPy archive."""

    output = Path(path)
    if output.suffix != ".npz":
        raise ValueError("matrix cache path must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite matrix cache: {output}")
    strings: dict[str, np.ndarray] = {}
    for name, values in (
        ("prompt_keys", matrix.prompt_keys),
        ("datasets", matrix.datasets),
        ("source_indices", matrix.source_indices),
        ("prompts", matrix.prompts),
        ("models", matrix.models),
    ):
        encoded, offsets = _encode_strings(values)
        strings[f"{name}_utf8"] = encoded
        strings[f"{name}_offsets"] = offsets
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(
            handle,
            schema=np.asarray(MATRIX_FILE_SCHEMA),
            **strings,
            quality=matrix.quality,
            realized_cost=matrix.realized_cost,
            prompt_tokens=matrix.prompt_tokens,
            completion_tokens=matrix.completion_tokens,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def load_outcome_matrix(path: Path) -> OutcomeMatrix:
    """Load and validate a pickle-free cached outcome matrix."""

    try:
        with np.load(Path(path), allow_pickle=False) as values:
            if str(values["schema"].item()) != MATRIX_FILE_SCHEMA:
                raise ValueError("matrix file schema does not match current loader")
            return OutcomeMatrix(
                prompt_keys=_decode_strings(values, "prompt_keys"),
                datasets=_decode_strings(values, "datasets"),
                source_indices=_decode_strings(values, "source_indices"),
                prompts=_decode_strings(values, "prompts"),
                models=_decode_strings(values, "models"),
                quality=np.asarray(values["quality"], dtype=np.float32),
                realized_cost=np.asarray(values["realized_cost"], dtype=np.float64),
                prompt_tokens=np.asarray(values["prompt_tokens"], dtype=np.int64),
                completion_tokens=np.asarray(values["completion_tokens"], dtype=np.int64),
            )
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"cannot load matrix cache: {error}") from error


def _encode_strings(values: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    blob = bytearray()
    offsets = [0]
    for value in values:
        blob.extend(value.encode("utf-8"))
        offsets.append(len(blob))
    return (
        np.frombuffer(blob, dtype=np.uint8).copy(),
        np.asarray(offsets, dtype="<u8"),
    )


def _decode_strings(values: Any, name: str) -> tuple[str, ...]:
    encoded = np.asarray(values[f"{name}_utf8"], dtype=np.uint8)
    offsets = np.asarray(values[f"{name}_offsets"], dtype=np.uint64)
    if encoded.ndim != 1 or offsets.ndim != 1 or len(offsets) == 0:
        raise ValueError(f"invalid {name} string encoding")
    if offsets[0] != 0 or offsets[-1] != len(encoded) or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"invalid {name} string offsets")
    try:
        return tuple(
            encoded[int(start) : int(end)].tobytes().decode("utf-8")
            for start, end in pairwise(offsets)
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid UTF-8 in {name}") from error


def benchmark_selection_digest(root: Path, paths: Iterable[Path]) -> str:
    """Bind the loader revision, requested roster, and selected archive paths."""

    source = Path(root).resolve()
    relative_paths: list[str] = []
    for path in sorted((Path(value).resolve() for value in paths), key=str):
        try:
            relative_paths.append(path.relative_to(source).as_posix())
        except ValueError as error:
            raise ValueError(f"selected file is outside benchmark root: {path}") from error
    payload = {
        "datasets": FLAGSHIP_DATASETS,
        "files": relative_paths,
        "loader_revision": MATRIX_LOADER_REVISION,
        "models": FLAGSHIP_MODELS,
        "splits": FLAGSHIP_SPLITS,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def save_prepared_matrix_cache(
    matrix: OutcomeMatrix,
    matrix_path: Path,
    manifest_path: Path,
    *,
    archive_sha256: str,
    selection_sha256: str,
    audit: LoadAudit,
) -> dict[str, Any]:
    """Write a normalized matrix and the source identity required to reuse it."""

    _require_digest(archive_sha256, "archive SHA-256")
    _require_digest(selection_sha256, "selection SHA-256")
    manifest_file = Path(manifest_path)
    if manifest_file.exists():
        raise FileExistsError(f"refusing to overwrite matrix manifest: {manifest_file}")
    save_outcome_matrix(matrix, matrix_path)
    manifest: dict[str, Any] = {
        "schema": MATRIX_CACHE_SCHEMA,
        "loader_revision": MATRIX_LOADER_REVISION,
        "archive_sha256": archive_sha256,
        "selection_sha256": selection_sha256,
        "matrix_sha256": outcome_matrix_digest(matrix),
        "models": list(matrix.models),
        "audit": {
            "files": audit.files,
            "raw_records": audit.raw_records,
            "accepted_prompts": audit.accepted_prompts,
            "rejections": dict(audit.rejections),
            "normalizations": dict(audit.normalizations),
        },
    }
    _write_json_atomic(manifest_file, manifest)
    return manifest


def load_prepared_matrix_cache(
    matrix_path: Path,
    manifest_path: Path,
    *,
    archive_sha256: str,
    selection_sha256: str | None = None,
    expected_models: tuple[str, ...] | None = None,
) -> tuple[OutcomeMatrix, dict[str, Any]]:
    """Load a matrix only when its cache manifest matches current inputs."""

    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read matrix cache manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("matrix cache manifest must be an object")
    if manifest.get("schema") != MATRIX_CACHE_SCHEMA:
        raise ValueError("matrix cache schema does not match current loader")
    if manifest.get("loader_revision") != MATRIX_LOADER_REVISION:
        raise ValueError("matrix cache loader revision does not match current loader")
    if manifest.get("archive_sha256") != archive_sha256:
        raise ValueError("matrix cache archive SHA-256 does not match pinned source")
    if selection_sha256 is not None and manifest.get("selection_sha256") != selection_sha256:
        raise ValueError("matrix cache selection SHA-256 does not match selected files")
    matrix = load_outcome_matrix(matrix_path)
    models = tuple(manifest.get("models", ()))
    if models != matrix.models:
        raise ValueError("matrix cache model roster does not match manifest")
    if expected_models is not None and matrix.models != expected_models:
        raise ValueError("matrix cache model roster does not match current requested roster")
    actual_digest = outcome_matrix_digest(matrix)
    if manifest.get("matrix_sha256") != actual_digest:
        raise ValueError("matrix SHA-256 does not match cache manifest")
    if not isinstance(manifest.get("audit"), dict):
        raise ValueError("matrix cache audit must be an object")
    return matrix, manifest


def _prompt_string(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list | dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    raise ValueError("prompt must be a string, list, or object")


def _require_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _write_json_atomic(path: Path, value: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        suffix=".json",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def _file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    return digest.hexdigest(), size


def _latest_file(paths: list[Path]) -> Path:
    def rank(path: Path) -> tuple[str, int]:
        stem = path.stem
        timestamp = stem[-15:] if len(stem) >= 15 else ""
        return timestamp, path.stat().st_mtime_ns

    return max(paths, key=rank)


def _benchmark_identity(path: Path) -> tuple[str, str, str]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return str(payload["dataset_name"]), str(payload["split"]), str(payload["model_name"])
