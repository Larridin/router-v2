"""
Drop-in replacement for AvengersPro's EmbeddingCache that uses local sentence-transformers
instead of making API calls. Same interface (get, batch) so the router code works unchanged.

Uses all-MiniLM-L6-v2 by default (384-dim, 80 MB, fast on CPU).
"""

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import List
import os

from sentence_transformers import SentenceTransformer
from loguru import logger


class LocalEmbeddingCache:
    """Thin wrapper around sentence-transformers with SQLite caching."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str | os.PathLike = ".cache",
        device: str = "cpu",
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self._model = SentenceTransformer(
            model_name, device=device, trust_remote_code=trust_remote_code
        )
        logger.info(f"Model loaded. Dimension: {self._model.get_sentence_embedding_dimension()}")

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self.db_path = cache_path / "local_embeddings.db"
        self._conn = self._open_conn()
        self._init_db()
        self._w_lock = threading.Lock()

    def get(self, text: str) -> List[float]:
        """Return the embedding for *text*, fetching from cache or computing locally."""
        text_hash = hashlib.md5(text.encode()).hexdigest()

        row = self._select(text_hash)
        if row is not None:
            return row

        with self._w_lock:
            row = self._select(text_hash)
            if row is not None:
                return row
            emb = self._model.encode(text, normalize_embeddings=True).tolist()
            self._insert(text_hash, text, emb)
            return emb

    def batch(self, texts: List[str], max_batch_size: int = 100) -> List[List[float]]:
        """Return embeddings for a list of texts (keeps order)."""
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), max_batch_size):
            chunk = texts[i : i + max_batch_size]

            hits: List[List[float]] = []
            misses: List[str] = []
            mapping: dict[str, int] = {}

            for idx, t in enumerate(chunk):
                h = hashlib.md5(t.encode()).hexdigest()
                row = self._select(h)
                if row is not None:
                    hits.append(row)
                else:
                    mapping[t] = idx
                    misses.append(t)

            if misses:
                embs = self._model.encode(
                    misses, normalize_embeddings=True, show_progress_bar=False
                )
                for text, emb in zip(misses, embs):
                    emb_list = emb.tolist()
                    h = hashlib.md5(text.encode()).hexdigest()
                    self._insert(h, text, emb_list)
                    hits.insert(mapping[text], emb_list)

            all_embeddings.extend(hits)

        return all_embeddings

    def _open_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path, timeout=30, check_same_thread=False, isolation_level=None
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    text_hash TEXT,
                    model     TEXT,
                    embedding TEXT NOT NULL,
                    text      TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(text_hash, model)
                )
                """
            )

    def _select(self, text_hash: str) -> List[float] | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT embedding FROM embeddings WHERE text_hash=? AND model=?",
                (text_hash, self.model_name),
            ).fetchone()
            if row:
                return json.loads(row[0])
            return None

    def _insert(self, text_hash: str, text: str, embedding: List[float]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (text_hash, model, embedding, text) VALUES (?,?,?,?)",
                (text_hash, self.model_name, json.dumps(embedding), text),
            )