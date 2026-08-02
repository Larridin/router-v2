"""LLMRouterBench bundle: save/load/route with flagship model roster + litellm pricing.

    python -m routerlab.llm_bundle save
    python -m routerlab.llm_bundle route "Explain quicksort in Python" --lam 30
    python -m routerlab.llm_bundle route "Explain quicksort in Python" --lam 0   # pure quality
"""
import json
import os
import sys
import time
import urllib.request
from functools import lru_cache

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from . import llm_data
from .embedder import embed_cached, get_embedder

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
VERSION = "v0.3"
LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

SEED = 42
MAX_PROMPT_CHARS = 1024

# ---- litellm pricing ----

# Model name mapping from our roster to litellm keys
LITELLM_MAP = {
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "deepseek-v3-0324": "deepseek/deepseek-chat",
    "deepseek-v3.1-terminus": "deepseek/deepseek-chat",
    "deepseek-r1-0528": "deepseek/deepseek-reasoner",
    "gemini-2.5-flash": "gemini/gemini-2.5-flash",
    "gemini-2.5-pro": "gemini/gemini-2.5-pro",
    "gpt-5-chat": "gpt-5",
    "gpt-5": "gpt-5",
    "qwen3-235b-a22b-2507": "deepinfra/Qwen/Qwen3-235B-A22B-Instruct-2507",
    "qwen3-235b-a22b-thinking-2507": "deepinfra/Qwen/Qwen3-235B-A22B-Thinking-2507",
    "glm-4.6": "cerebras/zai-glm-4.6",
    "kimi-k2-0905": "azure_ai/kimi-k2.5",
    "intern-s1": "deepseek/deepseek-chat",  # no exact litellm match; proxy
}


@lru_cache(maxsize=1)
def _fetch_litellm_prices() -> dict:
    """Fetch and cache litellm pricing JSON."""
    with urllib.request.urlopen(LITELLM_URL) as resp:
        return json.loads(resp.read())


def get_price_1k(model_name: str) -> tuple[float, float]:
    """Return (input_price_per_1k_tokens, output_price_per_1k_tokens) in USD."""
    key = LITELLM_MAP.get(model_name)
    if not key:
        return 0.001, 0.002  # sensible default
    prices = _fetch_litellm_prices()
    entry = prices.get(key, {})
    inp = float(entry.get("input_cost_per_token", 0) or 0)
    out = float(entry.get("output_cost_per_token", 0) or 0)
    return inp * 1000, out * 1000


# ---- prompt cost model ----

class PromptCostModel:
    """Estimate per-model cost for a prompt using litellm prices + token estimation."""

    CHARS_PER_TOKEN = 3.5  # rough: ~4 chars per token for English text

    def __init__(self, roster: list[str]):
        self.roster = roster
        self._prices = {m: get_price_1k(m) for m in roster}

    def predict(self, texts: list[str], output_tokens: int = 512) -> np.ndarray:
        """Return (n, M) cost matrix in USD."""
        out = np.zeros((len(texts), len(self.roster)))
        for i, t in enumerate(texts):
            input_tokens = max(1, len(t) / self.CHARS_PER_TOKEN)
            for j, model in enumerate(self.roster):
                inp_price, out_price = self._prices[model]
                out[i, j] = (input_tokens * inp_price + output_tokens * out_price) / 1000.0
        return out


# ---- estimator inlines (mirrors crossbench.py + estimators.py) ----

def _zscore(q):
    mu, sd = q.mean(axis=1, keepdims=True), q.std(axis=1, keepdims=True)
    out = np.zeros_like(q)
    np.divide(q - mu, sd, out=out, where=sd > 0)
    return out


def _shrink(mean, n, global_mean, k0):
    n = np.asarray(n, dtype=np.float64)[:, None]
    return (n * mean + k0 * global_mean[None, :]) / (n + k0)


_CODE_RE = __import__("re").compile(r"```|\bdef\b|\bclass\b|\breturn\b|\bimport\b|[{};]")
_MATH_RE = __import__("re").compile(r"[=+\-*/^<>]|\\frac|\\sum|\d+\.\d+")


def _text_features(texts):
    out = np.zeros((len(texts), 12), dtype=np.float32)
    for i, t in enumerate(texts):
        n = max(len(t), 1)
        words = t.split()
        out[i] = [
            len(t) / 1000.0, np.log1p(len(t)), t.count("\n") + 1, len(words),
            sum(len(w) for w in words) / max(len(words), 1),
            sum(c.isdigit() for c in t) / n, sum(ord(c) > 127 for c in t) / n,
            sum(not c.isalnum() and not c.isspace() for c in t) / n,
            sum(c.isupper() for c in t) / n,
            len(_CODE_RE.findall(t)) / n * 1000, len(_MATH_RE.findall(t)) / n * 1000,
            1.0 if t.rstrip().endswith("?") else 0.0,
        ]
    return out


class ClusterEst:
    def __init__(self, k=64, top_p=4, k0=10.0):
        self.k, self.top_p, self.k0 = k, top_p, k0

    def fit(self, emb, quality, texts=None):
        z = _zscore(quality)
        km = KMeans(n_clusters=self.k, n_init=3, random_state=SEED).fit(emb)
        c = km.cluster_centers_
        self.centroids = c / np.clip(np.linalg.norm(c, axis=1, keepdims=True), 1e-12, None)
        means, counts = np.zeros((self.k, quality.shape[1])), np.zeros(self.k)
        for ci in range(self.k):
            m = km.labels_ == ci
            counts[ci] = m.sum()
            if counts[ci]:
                means[ci] = z[m].mean(axis=0)
        self.table = _shrink(means, counts, z.mean(axis=0), self.k0)
        return self

    def estimate(self, emb, texts=None):
        top = np.argpartition(-(emb @ self.centroids.T), self.top_p - 1, axis=1)[:, :self.top_p]
        return self.table[top].mean(axis=1)


class KNNEst:
    def __init__(self, k=64, sharp=2.0, k0=10.0):
        self.k, self.sharp, self.k0 = k, sharp, k0

    def fit(self, emb, quality, texts=None):
        self.emb = emb.astype(np.float32)
        self.z = _zscore(quality)
        self.gm = self.z.mean(axis=0)
        return self

    def estimate(self, emb, texts=None, batch=1024):
        out = np.empty((len(emb), self.z.shape[1]))
        for i in range(0, len(emb), batch):
            b = emb[i:i + batch].astype(np.float32)
            sims = b @ self.emb.T
            top = np.argpartition(-sims, self.k - 1, axis=1)[:, :self.k]
            w = np.clip(sims[np.arange(len(b))[:, None], top], 0, None) ** self.sharp
            ws = w.sum(axis=1, keepdims=True)
            local = (w[:, :, None] * self.z[top]).sum(axis=1) / np.clip(ws, 1e-12, None)
            neff = np.zeros(len(b))
            np.divide(ws[:, 0] ** 2, (w**2).sum(axis=1), out=neff, where=ws[:, 0] > 0)
            out[i:i + batch] = _shrink(local, neff, self.gm, self.k0)
        return out


class GBTEst:
    def fit(self, emb, quality, texts):
        X = np.hstack([emb, _text_features(texts)])
        self.models = []
        for m in range(quality.shape[1]):
            g = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.08, max_leaf_nodes=63,
                early_stopping=True, validation_fraction=0.1, random_state=SEED,
            )
            g.fit(X, quality[:, m])
            self.models.append(g)
        return self

    def estimate(self, emb, texts):
        X = np.hstack([emb, _text_features(texts)])
        return np.clip(np.column_stack([g.predict(X) for g in self.models]), 0, 1)


def _calibrate(est, cal_emb, cal_q, cal_texts):
    raw = est.estimate(cal_emb, cal_texts)
    isos = []
    for m in range(cal_q.shape[1]):
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso.fit(raw[:, m], cal_q[:, m])
        isos.append(iso)
    return isos


def _apply_cal(est, isos, emb, texts):
    raw = est.estimate(emb, texts)
    return np.column_stack([iso.predict(raw[:, m]) for m, iso in enumerate(isos)])


# ---- save ----

def save():
    t0 = time.time()
    train, _test = llm_data.load()
    emb_model = get_embedder()
    train_emb = embed_cached(emb_model, train.texts, DATA)

    rng = np.random.default_rng(7)
    idx = rng.permutation(len(train.texts))
    n_fit = int(0.8 * len(idx))
    fit_i, cal_i = np.sort(idx[:n_fit]), np.sort(idx[n_fit:])
    fit_t = [train.texts[j] for j in fit_i]
    cal_t = [train.texts[j] for j in cal_i]

    members = [ClusterEst(k=64), KNNEst(), GBTEst()]
    cals = []
    for m in members:
        m.fit(train_emb[fit_i], train.quality[fit_i], fit_t)
        cals.append(_calibrate(m, train_emb[cal_i], train.quality[cal_i], cal_t))
        m.fit(train_emb, train.quality, train.texts)
        print(f"fitted+calibrated {type(m).__name__} @ {time.time() - t0:.0f}s")

    cost_model = PromptCostModel(llm_data.LLM_MODELS)

    out = os.path.join(MODELS_DIR, VERSION)
    if os.path.exists(out):
        raise SystemExit(f"{out} exists — bump VERSION instead of overwriting")
    os.makedirs(out)
    joblib.dump(
        {"members": members, "cals": cals, "roster": llm_data.LLM_MODELS, "cost_model": None},
        os.path.join(out, "router.joblib"), compress=3,
    )
    manifest = {
        "version": VERSION,
        "algorithm": "calibrated ensemble (cluster64 + knn64 + gbt) x dollar-EV rule",
        "embedder": emb_model.name,
        "roster": llm_data.LLM_MODELS,
        "n_train": len(train.texts),
        "training_data": "LLMRouterBench flagship models, 7 coding datasets",
        "decision_rule": "argmax_m P(success|prompt,m) - lambda * cost_litellm(prompt,m)",
        "pricing_source": "litellm model_prices_and_context_window.json",
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(MODELS_DIR, "latest"), "w") as f:
        f.write(VERSION + "\n")
    size = os.path.getsize(os.path.join(out, "router.joblib")) / 1e6
    print(f"saved {out} ({size:.0f} MB) in {time.time() - t0:.0f}s; latest -> {VERSION}")


# ---- Router ----

class Router:
    """Load a bundle and route prompts. lam is the $-per-quality tradeoff knob."""

    def __init__(self, version: str | None = None):
        version = version or open(os.path.join(MODELS_DIR, "latest")).read().strip()
        blob = joblib.load(os.path.join(MODELS_DIR, version, "router.joblib"))
        self.members = blob["members"]
        self.cals = blob["cals"]
        self.roster = blob["roster"]
        self.embedder = get_embedder()
        self.cost_model = PromptCostModel(self.roster)
        self.version = version

    def route(self, texts: list[str], lam: float = 30.0):
        texts = [llm_data.tail_truncate(t) for t in texts]
        emb = self.embedder.embed(texts)

        # Ensemble: mean of calibrated estimates
        P = np.mean(
            [_apply_cal(m, c, emb, texts) for m, c in zip(self.members, self.cals)],
            axis=0,
        )
        chat = self.cost_model.predict(texts)
        utility = P - lam * chat
        choice = np.argmax(utility, axis=1)

        return [
            {
                "model": self.roster[c],
                "p_success": round(float(P[i, c]), 3),
                "est_cost_usd": round(float(chat[i, c]), 6),
                "lambda": lam,
            }
            for i, c in enumerate(choice)
        ]


# ---- CLI ----

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "save":
        save()
    elif len(sys.argv) > 2 and sys.argv[1] == "route":
        lam = float(sys.argv[sys.argv.index("--lam") + 1]) if "--lam" in sys.argv else 30.0
        t0 = time.time()
        r = Router()
        load_ms = (time.time() - t0) * 1000
        t1 = time.time()
        res = r.route([sys.argv[2]], lam=lam)[0]
        route_ms = (time.time() - t1) * 1000
        print(json.dumps({
            **res, "bundle": r.version,
            "load_ms": round(load_ms), "route_ms": round(route_ms, 1),
        }, indent=2))
    else:
        print(__doc__)