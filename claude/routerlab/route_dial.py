"""Answer 'what does dial 70 pick?' precisely, both dial semantics.

    uv run python -m routerlab.route_dial <prompt-file> <dial 0-100>

(a) alpha-blend: score = d/100 * qualityNorm + (1-d/100) * cheapness
(b) calibrated dollars-dial: dial spaced over lambda mix-change breakpoints
Also prints this prompt's personal lambda crossover segments.
"""

import sys

import numpy as np

from . import dataset
from .embedder import embed_cached, get_embedder
from .estimators import ClusterEstimator, EnsembleEstimator, GBTEstimator, KNNEstimator
from .labelmatrix import ROSTER
from .roster import load_matrix
from .rules import PromptCostModel


def main(path: str, dial: float):
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

    # (a) literal alpha blend at d/100
    a = dial / 100.0
    qn = (P - P.min()) / max(P.max() - P.min(), 1e-12)
    cn = cost.mean(axis=0)
    cn = (cn - cn.min()) / max(cn.max() - cn.min(), 1e-12)
    alpha_pick = int(np.argmax(a * qn + (1 - a) * (1 - cn)))
    print(f"(a) literal {dial:.0f}/{100-dial:.0f} blend  -> {short[alpha_pick]}")

    # (b) calibrated dollars dial: breakpoints over the whole matrix
    Pm = ens.estimate_calibrated(X, texts)
    chm = cm.predict(texts)
    lambdas = np.concatenate([[0.0], np.geomspace(0.01, 5000.0, 300)])
    sigs, keep = [], []
    for lam in lambdas:
        ch = np.argmax(Pm - lam * chm, axis=1)
        sigs.append(np.bincount(ch, minlength=len(ROSTER)).tobytes())
    bp = [0] + [j for j in range(1, len(lambdas)) if sigs[j] != sigs[j - 1]]
    j = bp[min(int(round((100 - dial) / 100 * (len(bp) - 1))), len(bp) - 1)]
    lam70 = lambdas[j]
    ev_pick = int(np.argmax(P - lam70 * chat))
    print(f"(b) calibrated dial {dial:.0f} -> lambda={lam70:.3g} -> {short[ev_pick]} "
          f"(P={P[ev_pick]:.3f}, ${chat[ev_pick]:.5f}/req)  [{len(bp)} breakpoints]")

    # This prompt's personal winner segments over lambda.
    print("\nthis prompt's winner by lambda:")
    prev = None
    for lam in lambdas:
        w = int(np.argmax(P - lam * chat))
        if w != prev:
            print(f"  lambda >= {lam:8.3g}: {short[w]}  (P={P[w]:.3f}, ${chat[w]:.5f})")
            prev = w


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]))
