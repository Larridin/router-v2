"""Pinned local prompt embeddings with a content-addressed cache."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

BGE_MODEL_ID = "BAAI/bge-small-en-v1.5"
BGE_SOURCE_REPO = "qdrant/bge-small-en-v1.5-onnx-q"
BGE_MODEL_REVISION = "52398278842ec682c6f32300af41344b1c0b0bb2"
BGE_DIMENSION = 384
_CACHE_SCHEMA = "routerlab-embeddings-v1"


class Encoder(Protocol):
    """Minimal local batch encoder contract."""

    model_id: str
    source_repo: str
    revision: str
    dimension: int

    def encode(self, texts: tuple[str, ...]) -> NDArray[np.float32]:
        """Encode texts in their supplied order."""

        raise NotImplementedError


@dataclass(slots=True)
class FastEmbedEncoder:
    """Local BGE-small encoder backed by FastEmbed ONNX inference."""

    cache_dir: Path
    threads: int | None = None
    model_id: str = BGE_MODEL_ID
    source_repo: str = BGE_SOURCE_REPO
    revision: str = BGE_MODEL_REVISION
    dimension: int = BGE_DIMENSION

    def encode(self, texts: tuple[str, ...]) -> NDArray[np.float32]:
        """Encode raw prompts with the pinned BGE-small model."""

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        from fastembed import TextEmbedding
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(
            repo_id=self.source_repo,
            revision=self.revision,
            allow_patterns=[
                "config.json",
                "model_optimized.onnx",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
            cache_dir=str(self.cache_dir),
        )

        model = TextEmbedding(
            model_name=self.model_id,
            cache_dir=str(self.cache_dir),
            threads=self.threads,
            specific_model_path=model_path,
        )
        rows = list(model.embed(list(texts)))
        vectors = np.asarray(rows, dtype=np.float32)
        if vectors.shape != (len(texts), self.dimension):
            raise ValueError(
                f"encoder returned shape {vectors.shape}, expected {(len(texts), self.dimension)}"
            )
        return vectors


class EmbeddingCache:
    """Cache normalized unique-prompt embeddings by all semantic inputs."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get_or_encode(
        self,
        prompts: tuple[str, ...],
        encoder: Encoder,
    ) -> NDArray[np.float32]:
        """Return normalized embeddings, preserving duplicate prompt rows."""

        unique = tuple(dict.fromkeys(prompts))
        prompt_digest = _prompts_digest(unique)
        cache_key = _cache_key(encoder, prompt_digest)
        tensor_path = self.root / f"{cache_key}.npy"
        manifest_path = self.root / f"{cache_key}.json"
        expected = {
            "schema": _CACHE_SCHEMA,
            "model_id": encoder.model_id,
            "source_repo": encoder.source_repo,
            "revision": encoder.revision,
            "dimension": encoder.dimension,
            "rows": len(unique),
            "prompts_sha256": prompt_digest,
        }

        vectors = self._read(tensor_path, manifest_path, expected)
        if vectors is None:
            vectors = _normalize(encoder.encode(unique), len(unique), encoder.dimension)
            self._write(tensor_path, manifest_path, vectors, expected)

        positions = {prompt: index for index, prompt in enumerate(unique)}
        restored = np.asarray([vectors[positions[prompt]] for prompt in prompts], dtype=np.float32)
        return restored

    def _read(
        self,
        tensor_path: Path,
        manifest_path: Path,
        expected: dict[str, object],
    ) -> NDArray[np.float32] | None:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for field, value in expected.items():
                if manifest.get(field) != value:
                    return None
            raw = tensor_path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != manifest.get("tensor_sha256"):
                return None
            with tensor_path.open("rb") as handle:
                vectors = np.load(handle, allow_pickle=False)
            return _normalize(
                np.asarray(vectors, dtype=np.float32),
                int(expected["rows"]),
                int(expected["dimension"]),
            )
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError, ValueError):
            return None

    def _write(
        self,
        tensor_path: Path,
        manifest_path: Path,
        vectors: NDArray[np.float32],
        expected: dict[str, object],
    ) -> None:
        with tempfile.NamedTemporaryFile(dir=self.root, suffix=".npy", delete=False) as handle:
            temporary_tensor = Path(handle.name)
            np.save(handle, vectors, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        raw = temporary_tensor.read_bytes()
        manifest = {
            **expected,
            "dtype": "float32",
            "tensor_sha256": hashlib.sha256(raw).hexdigest(),
        }
        with tempfile.NamedTemporaryFile(
            dir=self.root,
            suffix=".json",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary_manifest = Path(handle.name)
            json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_tensor, tensor_path)
        os.replace(temporary_manifest, manifest_path)


def _prompts_digest(prompts: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for prompt in prompts:
        encoded = prompt.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _cache_key(encoder: Encoder, prompts_digest: str) -> str:
    fields = (
        _CACHE_SCHEMA,
        encoder.model_id,
        encoder.source_repo,
        encoder.revision,
        str(encoder.dimension),
        prompts_digest,
    )
    return hashlib.sha256("\0".join(fields).encode()).hexdigest()


def _normalize(
    vectors: NDArray[np.float32],
    rows: int,
    dimension: int,
) -> NDArray[np.float32]:
    values = np.asarray(vectors, dtype=np.float32)
    expected = (rows, dimension)
    if values.shape != expected:
        raise ValueError(f"embedding shape {values.shape}, expected {expected}")
    if not np.all(np.isfinite(values)):
        raise ValueError("embeddings contain non-finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("embeddings contain a zero vector")
    return np.asarray(values / norms, dtype=np.float32)
