from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from routerlab.embeddings import (
    BGE_MODEL_REVISION,
    BGE_SOURCE_REPO,
    EmbeddingCache,
    FastEmbedEncoder,
)


@dataclass
class FakeEncoder:
    model_id: str = "fake-encoder"
    source_repo: str = "fake/source"
    revision: str = "revision-1"
    dimension: int = 4
    calls: list[tuple[str, ...]] = field(default_factory=list)

    def encode(self, texts: tuple[str, ...]) -> np.ndarray:
        self.calls.append(texts)
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            rows.append([float(byte + 1) for byte in digest[: self.dimension]])
        return np.asarray(rows, dtype=np.float32)


class ExplodingEncoder(FakeEncoder):
    def encode(self, texts: tuple[str, ...]) -> np.ndarray:
        raise AssertionError(f"cache miss for {texts}")


def test_embedding_cache_encodes_unique_prompts_and_restores_duplicate_rows(
    tmp_path: Path,
) -> None:
    cache = EmbeddingCache(tmp_path)
    encoder = FakeEncoder()

    vectors = cache.get_or_encode(("alpha", "beta", "alpha"), encoder)

    assert encoder.calls == [("alpha", "beta")]
    assert vectors.shape == (3, 4)
    assert vectors.dtype == np.float32
    np.testing.assert_array_equal(vectors[0], vectors[2])
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), np.ones(3), atol=1e-6)


def test_embedding_cache_second_read_does_not_invoke_encoder(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    prompts = ("alpha", "beta")
    first = cache.get_or_encode(prompts, FakeEncoder())

    second = cache.get_or_encode(prompts, ExplodingEncoder())

    np.testing.assert_array_equal(first, second)


def test_embedding_cache_key_changes_with_model_revision(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    prompts = ("alpha",)
    first = FakeEncoder(revision="revision-1")
    second = FakeEncoder(revision="revision-2")
    third = FakeEncoder(revision="revision-1", source_repo="other/source")

    cache.get_or_encode(prompts, first)
    cache.get_or_encode(prompts, second)
    cache.get_or_encode(prompts, third)

    assert first.calls == [("alpha",)]
    assert second.calls == [("alpha",)]
    assert third.calls == [("alpha",)]
    assert len(list(tmp_path.glob("*.json"))) == 3
    assert len(list(tmp_path.glob("*.npy"))) == 3


def test_embedding_cache_recomputes_when_tensor_is_corrupt(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    prompts = ("alpha", "beta")
    original = FakeEncoder()
    cache.get_or_encode(prompts, original)
    tensor = next(tmp_path.glob("*.npy"))
    tensor.write_bytes(b"corrupt")
    replacement = FakeEncoder()

    recovered = cache.get_or_encode(prompts, replacement)

    assert replacement.calls == [("alpha", "beta")]
    assert recovered.shape == (2, 4)


def test_fastembed_download_is_pinned_to_declared_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    download_arguments: dict[str, object] = {}
    model_arguments: dict[str, object] = {}

    def snapshot_download(**kwargs: object) -> str:
        download_arguments.update(kwargs)
        return str(tmp_path / "snapshot")

    class TextEmbedding:
        def __init__(self, **kwargs: object) -> None:
            model_arguments.update(kwargs)

        def embed(self, texts: list[str]):
            return iter(np.ones((len(texts), 384), dtype=np.float32))

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)
    monkeypatch.setattr("fastembed.TextEmbedding", TextEmbedding)

    vectors = FastEmbedEncoder(tmp_path).encode(("prompt",))

    assert vectors.shape == (1, 384)
    assert download_arguments["repo_id"] == BGE_SOURCE_REPO
    assert download_arguments["revision"] == BGE_MODEL_REVISION
    assert model_arguments["specific_model_path"] == str(tmp_path / "snapshot")
