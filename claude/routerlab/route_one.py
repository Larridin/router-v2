"""Route one prompt through the current-roster router at several dial points.

    uv run python -m routerlab.route_one <prompt-file>

Fits the 7-model roster router from data/label_matrix.parquet (small, ~1 min),
then prints per-model calibrated P(success), predicted cost, and the pick at
several lambda dial positions.
"""

import sys

import numpy as np

from . import dataset
from .embedder import embed_cached, get_embedder
from .estimators import ClusterEstimator, EnsembleEstimator, GBTEstimator, KNNEstimator
from .labelmatrix import ROSTER
from .roster import load_matrix
from .rules import PromptCostModel

DIAL = [(0.0, "dial 100 (max quality)"), (5.0, "dial ~80"), (30.0, "dial ~60"),
        (150.0, "dial ~40"), (1000.0, "dial ~15 (max savings)")]


def main(path: str):
    prompt = open(path).read()
    texts, quality, cost, source = load_matrix()

    emb = get_embedder()
    X = embed_cached(emb, texts, str(dataset.__file__).rsplit("/", 2)[0] + "/data")
    rng = np.random.default_rng(7)
    p = rng.permutation(len(texts))
    fit_i, cal_i = np.sort(p[: int(0.8 * len(p))]), np.sort(p[int(0.8 * len(p)) :])
    t = lambda idx: [texts[j] for j in idx]

    members = [ClusterEstimator(k=8), KNNEstimator(k=16), GBTEstimator()]
    for m in members:
        m.fit(X[fit_i], quality[fit_i], t(fit_i))
        m.calibrate(X[cal_i], quality[cal_i], t(cal_i))
        m.fit(X, quality, texts)
    ens = EnsembleEstimator(members)
    cm = PromptCostModel().fit(texts, cost)

    q_text = dataset.tail_truncate(prompt)
    v = emb.embed([q_text])
    P = ens.estimate_calibrated(v, [q_text])[0]
    chat = cm.predict([q_text])[0]

    short = [m.split("/")[-1] for m in ROSTER]
    order = np.argsort(-P)
    print(f"\nprompt: {len(prompt)} chars, embedded tail {len(q_text)} chars\n")
    print(f"{'model':28s} {'P(success)':>10s} {'est cost/req':>13s}")
    for j in order:
        print(f"{short[j]:28s} {P[j]:>10.3f} {chat[j]:>12.5f}")
    print()
    for lam, label in DIAL:
        pick = int(np.argmax(P - lam * chat))
        print(f"{label:26s} -> {short[pick]:28s} (P={P[pick]:.3f}, ${chat[pick]:.5f}/req)")


if __name__ == "__main__":
    main(sys.argv[1])
