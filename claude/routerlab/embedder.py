"""Sentence embedder: MiniLM-L6-v2 via ONNX Runtime (mean-pool + L2 norm).

Falls back to TF-IDF + TruncatedSVD if the ONNX assets are unavailable, so the
pipeline stays runnable offline. Both routers always share one embedder, so the
comparison stays fair either way.
"""

import hashlib
import os

import numpy as np

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "minilm")
HF_BASE = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
MAX_TOKENS = 256


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


class OnnxMiniLM:
    name = "minilm-l6-v2-onnx"

    def __init__(self, asset_dir: str = ASSET_DIR):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(os.path.join(asset_dir, "tokenizer.json"))
        self.tok.enable_truncation(max_length=MAX_TOKENS)
        self.tok.enable_padding()
        self.sess = ort.InferenceSession(
            os.path.join(asset_dir, "model.onnx"), providers=["CPUExecutionProvider"]
        )

    def embed(self, texts: list[str], batch: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            encs = self.tok.encode_batch(texts[i : i + batch])
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if any(inp.name == "token_type_ids" for inp in self.sess.get_inputs()):
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self.sess.run(None, feeds)[0]  # (b, seq, 384)
            m = mask[:, :, None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            out.append(pooled)
        return _l2(np.concatenate(out).astype(np.float32))


class TfidfSvd:
    name = "tfidf-svd-384"

    def __init__(self):
        self._fitted = None

    def embed(self, texts: list[str], batch: int = 0) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        if self._fitted is None:
            vec = TfidfVectorizer(max_features=50000, sublinear_tf=True, analyzer="char_wb", ngram_range=(2, 4))
            X = vec.fit_transform(texts)
            svd = TruncatedSVD(n_components=384, random_state=42).fit(X)
            self._fitted = (vec, svd)
        vec, svd = self._fitted
        return _l2(svd.transform(vec.transform(texts)).astype(np.float32))


def get_embedder():
    try:
        return OnnxMiniLM()
    except Exception as e:  # missing assets, bad download, etc.
        print(f"[embedder] ONNX MiniLM unavailable ({e}); falling back to TF-IDF+SVD")
        return TfidfSvd()


def embed_cached(emb, texts: list[str], cache_dir: str) -> np.ndarray:
    """Embed with an on-disk cache keyed by embedder name + corpus hash."""
    h = hashlib.sha256(("\x00".join(texts) + emb.name).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"emb_{emb.name}_{h}.npy")
    if os.path.exists(path):
        return np.load(path)
    vecs = emb.embed(texts)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(path, vecs)
    return vecs
