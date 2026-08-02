"""Robustness checks on the kNN-vs-cluster result. Two experiments:

A. Sensitivity: does the cluster router close the gap with more clusters
   (finer quantization)? Is the kNN win stable across K/sharpness?
B. Leave-one-benchmark-family-out (LOBO): train with a whole task family held
   out, test on it — generalization vs benchmark memorization.

Reuses the standard split's cached embeddings; assembles the full-corpus
matrix from them without re-embedding.
"""

import json
import os
import time

import numpy as np

from . import core, dataset, evals
from .baseline import ClusterRouter
from .embedder import embed_cached, get_embedder
from .knn_router import KNNRouter

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

BUDGETS_PER_1K = [0.25, 1.0, 2.0]  # $ per 1k prompts


def family(src: str) -> str:
    s = src.lower()
    if s.startswith("mmlu"):
        return "mmlu"
    if s.startswith("chinese") or s.startswith("chinese-"):
        return "chinese"
    if s.startswith("mtbench"):
        return "mtbench"
    return {
        "hellaswag": "hellaswag",
        "grade-school-math": "gsm8k",
        "arc-challenge": "arc",
        "winogrande": "winogrande",
        "mbpp": "mbpp",
    }.get(s, "misc")


LOBO_FAMILIES = ["mmlu", "hellaswag", "gsm8k", "arc", "winogrande", "mbpp", "mtbench", "chinese"]


def frontier_metrics(router, Q, quality, cost, refs) -> dict:
    points = evals.sweep(router, Q, quality, cost, dataset.MODELS, refs)
    out = {f"q@${b}/1k": evals.quality_at_cost(points, b / 1000) for b in BUDGETS_PER_1K}
    best = max(points, key=lambda p: p.quality)
    out["peak_q"] = best.quality
    out["peak_cost_per_1k"] = best.cost * 1000
    out["gap_recovered_at_peak"] = best.gap_recovered
    return out


def full_corpus_embeddings(emb, train, test):
    """Reassemble the full-corpus embedding matrix from the split caches."""
    tr = embed_cached(emb, train.texts, DATA)  # cache hits from run.py
    te = embed_cached(emb, test.texts, DATA)
    texts = train.texts + test.texts
    X = np.concatenate([tr, te])
    quality = np.concatenate([train.quality, test.quality])
    cost = np.concatenate([train.cost, test.cost])
    source = np.concatenate([train.source, test.source])
    return texts, X, quality, cost, source


def sensitivity(train_emb, train_q, train_c, test_emb, test_q, test_c, refs) -> list[dict]:
    rows = []
    for k in [8, 16, 32, 64, 128, 256]:
        t = time.time()
        # n_init=10 gets slow for large k; 3 restarts is enough for a sensitivity read.
        r = ClusterRouter(k=k, n_init=10 if k <= 32 else 3)
        r.fit(train_emb, train_q, train_c)
        m = frontier_metrics(r, r.precompute(test_emb), test_q, test_c, refs)
        rows.append({"router": f"cluster k={k}", **m, "fit_s": round(time.time() - t, 1)})
        print(rows[-1])
    for kn in [16, 32, 64, 128, 256]:
        for sharp in [2.0, 8.0, 32.0]:
            t = time.time()
            r = KNNRouter(k=kn, sharpness=sharp).fit(train_emb, train_q, train_c)
            m = frontier_metrics(r, r.precompute(test_emb), test_q, test_c, refs)
            rows.append({"router": f"knn K={kn} sharp={sharp:g}", **m, "fit_s": round(time.time() - t, 1)})
            print(rows[-1])
    return rows


def lobo(X, quality, cost, fams) -> list[dict]:
    rows = []
    for held in LOBO_FAMILIES:
        test_mask = fams == held
        train_mask = ~test_mask
        refs = evals.references(quality[test_mask], cost[test_mask], dataset.MODELS)
        row = {
            "held_out": held,
            "n_test": int(test_mask.sum()),
            "oracle_q": refs["oracle"]["quality"],
            "best_single_q": refs["singles"][refs["best_single"]]["quality"],
            "best_single": refs["best_single"],
        }
        for r in [ClusterRouter(), KNNRouter()]:
            r.fit(X[train_mask], quality[train_mask], cost[train_mask])
            Q = r.precompute(X[test_mask])
            # Pure-quality operating point: quality-estimate transfer, no cost dial.
            ch = evals.choose(Q, r.cost_norm, r.cost_scale, alpha=1.0)
            rq, rc = evals.realized(ch, quality[test_mask], cost[test_mask])
            tag = "cluster" if isinstance(r, ClusterRouter) else "knn"
            row[f"{tag}_q_alpha1"] = float(rq.mean())
            row[f"{tag}_cost_alpha1_per_1k"] = float(rc.mean() * 1000)
        print(row)
        rows.append(row)
    return rows


def main():
    t0 = time.time()
    train, test = dataset.load(os.path.join(DATA, "routerbench_0shot.pkl"))
    emb = get_embedder()
    texts, X, quality, cost, source = full_corpus_embeddings(emb, train, test)
    fams = np.array([family(s) for s in source])
    print(f"corpus {X.shape}, families: {dict(zip(*np.unique(fams, return_counts=True)))}")

    n_train = len(train.texts)
    train_emb, test_emb = X[:n_train], X[n_train:]
    refs = evals.references(test.quality, test.cost, dataset.MODELS)

    print("\n=== A. sensitivity (standard split) ===")
    sens = sensitivity(train_emb, train.quality, train.cost, test_emb, test.quality, test.cost, refs)

    print("\n=== B. leave-one-family-out ===")
    lob = lobo(X, quality, cost, fams)

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "robustness.json"), "w") as f:
        json.dump({"sensitivity": sens, "lobo": lob}, f, indent=2)
    report(sens, lob)
    print(f"\ndone in {time.time()-t0:.0f}s -> results/robustness.json, robustness.md")


def report(sens, lob):
    lines = ["# Robustness checks", "", "## A. Sensitivity (held-out split, quality at matched cost)", ""]
    hdr = "| config | " + " | ".join(f"q@${b}/1k" for b in BUDGETS_PER_1K) + " | peak q | peak $/1k |"
    lines += [hdr, "|" + "---|" * (len(BUDGETS_PER_1K) + 3)]
    for r in sens:
        cells = " | ".join("—" if r[f"q@${b}/1k"] != r[f"q@${b}/1k"] else f"{r[f'q@${b}/1k']:.3f}" for b in BUDGETS_PER_1K)
        lines.append(f"| {r['router']} | {cells} | {r['peak_q']:.3f} | {r['peak_cost_per_1k']:.2f} |")
    lines += ["", "## B. Leave-one-family-out (alpha=1, pure quality routing)", ""]
    lines += ["| held-out family | n | oracle | best single | cluster q | knn q | knn − cluster |", "|" + "---|" * 7]
    for r in lob:
        d = r["knn_q_alpha1"] - r["cluster_q_alpha1"]
        lines.append(
            f"| {r['held_out']} | {r['n_test']} | {r['oracle_q']:.3f} | {r['best_single_q']:.3f} ({r['best_single'].split('/')[-1]}) "
            f"| {r['cluster_q_alpha1']:.3f} | {r['knn_q_alpha1']:.3f} | {d:+.3f} |"
        )
    with open(os.path.join(RESULTS, "robustness.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
