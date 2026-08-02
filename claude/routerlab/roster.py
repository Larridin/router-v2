"""Train + eval the router on OUR roster's label matrix (data/label_matrix.parquet).

    python -m routerlab.roster

Same algorithm, same eval harness as the RouterBench work; only the matrix
changed. Corpus is small (903 train / 387 test), so: cluster k=8 (>=100
prompts/cluster rule), and CIs are reported honestly (~±0.03).
"""

import json
import os

import numpy as np
import pandas as pd

from . import dataset, evals
from .embedder import embed_cached, get_embedder
from .estimators import ClusterEstimator, EnsembleEstimator, GBTEstimator, KNNEstimator
from .labelmatrix import CORPUS, ROSTER
from .rules import AlphaBlendRule, DollarEVRule, PromptCostModel

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
BUDGETS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]  # $/1k — this roster's cost range


def load_matrix():
    prompts = {json.loads(l)["id"]: json.loads(l) for l in open(CORPUS)}
    df = pd.read_parquet(os.path.join(DATA, "label_matrix.parquet"))
    piv_q = df.pivot(index="id", columns="model", values="quality")[ROSTER]
    piv_c = df.pivot(index="id", columns="model", values="cost")[ROSTER]
    ids = [i for i in piv_q.index if not piv_q.loc[i].isna().any()]
    texts = [dataset.tail_truncate(prompts[i]["prompt"]) for i in ids]
    source = np.array([prompts[i]["source"] for i in ids])
    return texts, piv_q.loc[ids].to_numpy(), piv_c.loc[ids].to_numpy(), source


def main():
    texts, quality, cost, source = load_matrix()
    rng = np.random.default_rng(42)
    test_mask = np.zeros(len(texts), dtype=bool)
    for src in np.unique(source):
        idx = np.flatnonzero(source == src)
        rng.shuffle(idx)
        test_mask[idx[: int(round(0.3 * len(idx)))]] = True
    tr, te = ~test_mask, test_mask
    print(f"matrix: {len(texts)} prompts x {len(ROSTER)} models; train {tr.sum()} / test {te.sum()}")

    emb = get_embedder()
    X = embed_cached(emb, texts, DATA)

    # Inner 80/20 for calibration.
    tr_i = np.flatnonzero(tr)
    p = np.random.default_rng(7).permutation(len(tr_i))
    fit_i, cal_i = tr_i[np.sort(p[: int(0.8 * len(p))])], tr_i[np.sort(p[int(0.8 * len(p)) :])]
    t = lambda idx: [texts[j] for j in idx]

    members = [ClusterEstimator(k=8), KNNEstimator(k=16), GBTEstimator()]
    for m in members:
        m.fit(X[fit_i], quality[fit_i], t(fit_i))
        m.calibrate(X[cal_i], quality[cal_i], t(cal_i))
        m.fit(X[tr_i], quality[tr_i], t(tr_i))
    ens = EnsembleEstimator(members)

    weave = ClusterEstimator(k=8)  # the Weave algorithm alone, corpus-scaled k
    weave.fit(X[fit_i], quality[fit_i]).calibrate(X[cal_i], quality[cal_i], t(cal_i))
    weave.fit(X[tr_i], quality[tr_i])

    te_i = np.flatnonzero(te)
    refs = evals.references(quality[te_i], cost[te_i], ROSTER)
    alpha_rule = AlphaBlendRule(cost[fit_i].mean(axis=0))
    ev_rule = DollarEVRule(PromptCostModel().fit(t(fit_i), cost[fit_i]))

    out = {"refs": refs, "frontiers": {}}
    for name, est in [("weave-algorithm (cluster k=8)", weave), ("ensemble (ours)", ens)]:
        P = est.estimate_calibrated(X[te_i], t(te_i))
        pts = []
        for rule in (alpha_rule, ev_rule):
            for prm in rule.params:
                ch = rule.choices(P, t(te_i), float(prm))
                rq, rc = evals.realized(ch, quality[te_i], cost[te_i])
                lo, hi = evals.bootstrap_ci(rq)
                pts.append({"param": float(prm), "rule": rule.name, "quality": float(rq.mean()),
                            "cost": float(rc.mean()), "q_lo": lo, "q_hi": hi})
        out["frontiers"][name] = pts

    with open(os.path.join(RESULTS, "roster_eval.json"), "w") as f:
        json.dump(out, f, indent=2)

    bs = refs["singles"][refs["best_single"]]
    lines = [
        "# Router on OUR roster (7 current models, own label matrix)",
        "",
        f"- n_test = {te.sum()} (CIs ≈ ±0.03 — small-corpus caveat)",
        f"- Oracle: {refs['oracle']['quality']:.3f} @ ${refs['oracle']['cost']*1000:.2f}/1k",
        f"- Best single: `{refs['best_single']}` {bs['quality']:.3f} @ ${bs['cost']*1000:.2f}/1k",
        f"- Cheapest: `{refs['cheapest']}` {refs['singles'][refs['cheapest']]['quality']:.3f} "
        f"@ ${refs['singles'][refs['cheapest']]['cost']*1000:.2f}/1k",
        "",
        "| router | " + " | ".join(f"q@${b}/1k" for b in BUDGETS) + " | peak | parity cost | savings |",
        "|" + "---|" * (len(BUDGETS) + 4),
    ]
    for name, pts in out["frontiers"].items():
        cells = []
        for b in BUDGETS:
            ok = [p["quality"] for p in pts if p["cost"] * 1000 <= b]
            cells.append(f"{max(ok):.3f}" if ok else "—")
        peak = max(p["quality"] for p in pts)
        parity = [p["cost"] for p in pts if p["quality"] >= bs["quality"]]
        pc = f"${min(parity)*1000:.2f}" if parity else "never"
        sav = f"{1 - min(parity)/bs['cost']:.0%}" if parity else "—"
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {peak:.3f} | {pc} | {sav} |")
    report = "\n".join(lines)
    with open(os.path.join(RESULTS, "roster_eval.md"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
