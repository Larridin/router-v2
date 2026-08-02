"""Persist the winning router as a loadable, versioned artifact.

    python -m routerlab.bundle save              # fit + write models/v0.1/
    python -m routerlab.bundle route "prompt..."  [--lam 30]

Bundle = calibrated {cluster256, knn64, gbt} ensemble + dollar-EV rule
(PromptCostModel + lambda knob), trained per the `final` protocol: members
fit on the 80% fit fold for calibration, isotonic on the 20% cal fold,
then refit on full train. Mirrors the Weave repo's artifact discipline:
versioned dir, manifest, never overwrite.
"""

import json
import os
import sys
import time

import joblib
import numpy as np

from . import dataset
from .embedder import embed_cached, get_embedder
from .estimators import ClusterEstimator, EnsembleEstimator, GBTEstimator, KNNEstimator
from .rules import PromptCostModel

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
VERSION = "v0.2"  # v0.1 shipped a floorless cost model: short prompts priced ~0, lambda inert


def save():
    t0 = time.time()
    train, test = dataset.load(os.path.join(DATA, "routerbench_0shot.pkl"))
    emb = get_embedder()
    train_emb = embed_cached(emb, train.texts, DATA)

    rng = np.random.default_rng(7)
    idx = rng.permutation(len(train.texts))
    n_fit = int(0.8 * len(idx))
    fit_i, cal_i = np.sort(idx[:n_fit]), np.sort(idx[n_fit:])
    fit_t = [train.texts[j] for j in fit_i]
    cal_t = [train.texts[j] for j in cal_i]

    members = [ClusterEstimator(), KNNEstimator(), GBTEstimator()]
    for m in members:
        m.fit(train_emb[fit_i], train.quality[fit_i], fit_t)
        m.calibrate(train_emb[cal_i], train.quality[cal_i], cal_t)
        m.fit(train_emb, train.quality, train.texts)
        print(f"fitted+calibrated {m.name} @ {time.time()-t0:.0f}s")
    cost_model = PromptCostModel().fit(train.texts, train.cost)

    out = os.path.join(MODELS_DIR, VERSION)
    if os.path.exists(out):
        raise SystemExit(f"{out} exists — bump VERSION instead of overwriting (artifact discipline)")
    os.makedirs(out)
    joblib.dump({"members": members, "cost_model": cost_model}, os.path.join(out, "router.joblib"), compress=3)
    manifest = {
        "version": VERSION,
        "algorithm": "calibrated ensemble (cluster256 + knn64 + gbt) x dollar-EV rule",
        "embedder": emb.name,
        "roster": dataset.MODELS,
        "n_train": len(train.texts),
        "trained_from": "routerbench_0shot.pkl, split seed 42, cal seed 7",
        "decision_rule": "argmax_m P(success|prompt,m) - lambda * cost_hat(prompt,m); lambda=0 pure quality",
        "reference_metrics": "see ../results/relative_gains.md and hypotheses.md (frontier: ensembleFT)",
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(MODELS_DIR, "latest"), "w") as f:
        f.write(VERSION + "\n")
    size = os.path.getsize(os.path.join(out, "router.joblib")) / 1e6
    print(f"saved {out} ({size:.0f} MB) in {time.time()-t0:.0f}s; latest -> {VERSION}")


class Router:
    """Load a bundle and route prompts. lam is the $-per-quality tradeoff knob."""

    def __init__(self, version: str | None = None):
        version = version or open(os.path.join(MODELS_DIR, "latest")).read().strip()
        blob = joblib.load(os.path.join(MODELS_DIR, version, "router.joblib"))
        self.ensemble = EnsembleEstimator(blob["members"])
        self.cost_model = blob["cost_model"]
        self.embedder = get_embedder()
        self.version = version

    def route(self, texts: list[str], lam: float = 30.0):
        texts = [dataset.tail_truncate(dataset.prompt_text(t)) for t in texts]
        emb = self.embedder.embed(texts)
        p = self.ensemble.estimate_calibrated(emb, texts)
        utility = p - lam * self.cost_model.predict(texts)
        choice = np.argmax(utility, axis=1)
        return [
            {
                "model": dataset.MODELS[c],
                "p_success": round(float(p[i, c]), 3),
                "est_cost_usd": round(float(self.cost_model.predict([texts[i]])[0, c]), 6),
                "lambda": lam,
            }
            for i, c in enumerate(choice)
        ]


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
        print(json.dumps({**res, "bundle": r.version, "load_ms": round(load_ms), "route_ms": round(route_ms, 1)}, indent=2))
    else:
        print(__doc__)
