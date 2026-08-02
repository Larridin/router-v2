import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pytest

from routerlab.data import (
    download_archive,
    extract_archive,
    load_outcome_files,
    load_outcome_matrix,
    load_prepared_matrix_cache,
    normalize_prompt,
    prompt_key,
    save_outcome_matrix,
    save_prepared_matrix_cache,
)
from routerlab.schema import OutcomeMatrix, outcome_matrix_digest

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_outcomes_builds_complete_shared_matrix() -> None:
    matrix, audit = load_outcome_files([FIXTURES / "tiny_outcomes.json"])

    assert matrix.models == ("model-a", "model-b", "model-c")
    assert matrix.prompts == ("Explain alpha.", "Solve beta.")
    assert matrix.datasets == ("demo", "demo")
    assert matrix.source_indices == ("1", "2")
    assert matrix.quality.dtype == np.float32
    assert matrix.realized_cost.dtype == np.float64
    np.testing.assert_allclose(
        matrix.quality,
        np.array([[0.5, 0.9, 0.2], [0.1, 0.4, 1.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        matrix.realized_cost,
        np.array([[0.01, 0.03, 0.02], [0.01, 0.03, 0.02]], dtype=np.float64),
    )
    assert audit.raw_records == 8
    assert audit.accepted_prompts == 2
    assert audit.rejections == {"incomplete_roster": 1}


def test_prompt_identity_normalizes_only_outer_whitespace_and_newlines() -> None:
    assert normalize_prompt("  one\r\ntwo \n") == "one\ntwo"
    assert prompt_key("  one\r\ntwo \n") == prompt_key("one\ntwo")
    assert prompt_key("one  two") != prompt_key("one two")


def test_loader_rejects_non_finite_outcome(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        '{"records":[{"dataset":"d","model":"m","index":1,'
        '"prompt":"p","score":NaN,"cost":1,"prompt_tokens":1,'
        '"completion_tokens":1}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-finite score"):
        load_outcome_files([path])


def test_loader_audits_negative_provider_token_sentinel(tmp_path: Path) -> None:
    path = tmp_path / "negative-token.json"
    path.write_text(
        '{"records":[{"dataset":"d","model":"m","index":1,'
        '"prompt":"p","score":0,"cost":0,"prompt_tokens":0,'
        '"completion_tokens":-52}]}',
        encoding="utf-8",
    )

    matrix, audit = load_outcome_files([path])

    assert matrix.completion_tokens.tolist() == [[0]]
    assert audit.normalizations == {"negative_completion_tokens_to_zero": 1}


def test_loader_inherits_real_benchmark_file_metadata(tmp_path: Path) -> None:
    paths: list[Path] = []
    for model, scores in (("model-a", [0.2, 0.7]), ("model-b", [0.9, 0.3])):
        path = tmp_path / f"{model}.json"
        path.write_text(
            json.dumps(
                {
                    "dataset_name": "aime",
                    "split": "test",
                    "model_name": model,
                    "records": [
                        {
                            "index": index,
                            "prompt": f"prompt {index}",
                            "score": score,
                            "cost": 0.1,
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                        }
                        for index, score in enumerate(scores)
                    ],
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    matrix, audit = load_outcome_files(paths)

    assert matrix.models == ("model-a", "model-b")
    assert matrix.datasets == ("aime", "aime")
    np.testing.assert_allclose(matrix.quality, [[0.2, 0.9], [0.7, 0.3]])
    assert audit.accepted_prompts == 2


def test_matrix_cache_round_trip_uses_no_pickle(tmp_path: Path) -> None:
    matrix, _ = load_outcome_files([FIXTURES / "tiny_outcomes.json"])
    path = tmp_path / "matrix.npz"

    save_outcome_matrix(matrix, path)
    loaded = load_outcome_matrix(path)

    assert loaded.prompt_keys == matrix.prompt_keys
    assert loaded.datasets == matrix.datasets
    assert loaded.prompts == matrix.prompts
    assert loaded.models == matrix.models
    np.testing.assert_array_equal(loaded.quality, matrix.quality)
    np.testing.assert_array_equal(loaded.realized_cost, matrix.realized_cost)


def test_matrix_cache_stores_variable_length_prompt_text(tmp_path: Path) -> None:
    prompts = ("x" * 16_384, *(f"short-{index}" for index in range(256)))
    rows = len(prompts)
    matrix = OutcomeMatrix(
        prompt_keys=tuple(prompt_key(prompt) for prompt in prompts),
        datasets=tuple("demo" for _ in prompts),
        source_indices=tuple(str(index) for index in range(rows)),
        prompts=prompts,
        models=("model-a",),
        quality=np.ones((rows, 1), dtype=np.float32),
        realized_cost=np.ones((rows, 1), dtype=np.float64),
        prompt_tokens=np.ones((rows, 1), dtype=np.int64),
        completion_tokens=np.ones((rows, 1), dtype=np.int64),
    )
    path = tmp_path / "matrix.npz"

    save_outcome_matrix(matrix, path)

    with zipfile.ZipFile(path) as archive:
        assert "prompts.npy" not in archive.namelist()
        assert archive.getinfo("prompts_utf8.npy").file_size < 100_000
    assert load_outcome_matrix(path).prompts == prompts


def test_outcome_matrix_digest_binds_metadata_and_outcomes() -> None:
    matrix, _ = load_outcome_files([FIXTURES / "tiny_outcomes.json"])
    changed_quality = matrix.quality.copy()
    changed_quality[0, 0] += 0.01
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

    assert outcome_matrix_digest(matrix) == outcome_matrix_digest(matrix)
    assert outcome_matrix_digest(matrix) != outcome_matrix_digest(changed)


def test_prepared_matrix_cache_validates_source_selection_and_content(tmp_path: Path) -> None:
    matrix, audit = load_outcome_files([FIXTURES / "tiny_outcomes.json"])
    matrix_path = tmp_path / "outcomes.npz"
    manifest_path = tmp_path / "outcomes.manifest.json"
    archive_sha256 = "a" * 64
    selection_sha256 = "b" * 64
    saved = save_prepared_matrix_cache(
        matrix,
        matrix_path,
        manifest_path,
        archive_sha256=archive_sha256,
        selection_sha256=selection_sha256,
        audit=audit,
    )

    loaded, manifest = load_prepared_matrix_cache(
        matrix_path,
        manifest_path,
        archive_sha256=archive_sha256,
        selection_sha256=selection_sha256,
    )

    assert outcome_matrix_digest(loaded) == outcome_matrix_digest(matrix)
    assert manifest == saved
    with pytest.raises(ValueError, match="archive"):
        load_prepared_matrix_cache(
            matrix_path,
            manifest_path,
            archive_sha256="c" * 64,
            selection_sha256=selection_sha256,
        )
    with pytest.raises(ValueError, match="selection"):
        load_prepared_matrix_cache(
            matrix_path,
            manifest_path,
            archive_sha256=archive_sha256,
            selection_sha256="d" * 64,
        )

    value = json.loads(manifest_path.read_text())
    value["matrix_sha256"] = "e" * 64
    manifest_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="matrix SHA-256"):
        load_prepared_matrix_cache(
            matrix_path,
            manifest_path,
            archive_sha256=archive_sha256,
            selection_sha256=selection_sha256,
        )


def test_download_verifies_digest_and_reuses_valid_archive(tmp_path: Path) -> None:
    payload = b"public benchmark bytes"
    digest = hashlib.sha256(payload).hexdigest()
    requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, content=payload, request=request)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        first = download_archive(
            tmp_path,
            client=client,
            url="https://example.test/bench.tar.gz",
            expected_sha256=digest,
        )
        second = download_archive(
            tmp_path,
            client=client,
            url="https://example.test/bench.tar.gz",
            expected_sha256=digest,
        )

    assert first.path.read_bytes() == payload
    assert not first.reused
    assert second.reused
    assert requests == 1


def test_extract_rejects_archive_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 4
        handle.addfile(info, io.BytesIO(b"nope"))

    with pytest.raises(ValueError, match="unsafe archive member"):
        extract_archive(archive, tmp_path / "output")
    assert not (tmp_path / "escape.txt").exists()
