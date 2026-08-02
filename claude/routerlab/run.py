"""End-to-end: load RouterBench, train both routers, eval on held-out split, report."""

import json
import os
import time

import numpy as np

from . import dataset, evals
from .baseline import ClusterRouter
from .embedder import embed_cached, get_embedder
from .knn_router import KNNRouter

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")


def main():
    t0 = time.time()
    train, test = dataset.load(os.path.join(DATA, "routerbench_0shot.pkl"))
    print(f"train={len(train.texts)} test={len(test.texts)} models={len(dataset.MODELS)}")

    emb = get_embedder()
    print(f"embedder: {emb.name}")
    train_emb = embed_cached(emb, train.texts, DATA)
    test_emb = embed_cached(emb, test.texts, DATA)
    print(f"embedded in {time.time()-t0:.0f}s: train {train_emb.shape}, test {test_emb.shape}")

    routers = [
        ClusterRouter().fit(train_emb, train.quality, train.cost),
        KNNRouter().fit(train_emb, train.quality, train.cost),
    ]

    refs = evals.references(test.quality, test.cost, dataset.MODELS)
    out = {"embedder": emb.name, "n_train": len(train.texts), "n_test": len(test.texts), "refs": refs, "routers": {}}

    for r in routers:
        t1 = time.time()
        Q = r.precompute(test_emb)
        points = evals.sweep(r, Q, test.quality, test.cost, dataset.MODELS, refs)
        out["routers"][r.name] = [vars(p) for p in points]
        best = max(points, key=lambda p: p.quality)
        print(f"{r.name}: swept {len(points)} alphas in {time.time()-t1:.0f}s; peak quality {best.quality:.3f} @ ${best.cost*1000:.2f}/1k prompts")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    plot(out)
    report(out)
    print(f"done in {time.time()-t0:.0f}s -> results/metrics.json, frontier.png, report.md")


def plot(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, pts in out["routers"].items():
        pts = sorted(pts, key=lambda p: p["cost"])
        ax.plot([p["cost"] * 1000 for p in pts], [p["quality"] for p in pts], marker="o", ms=3, label=name)
    for m, s in out["refs"]["singles"].items():
        ax.scatter(s["cost"] * 1000, s["quality"], color="gray", s=18, zorder=3)
        ax.annotate(m.split("/")[-1], (s["cost"] * 1000, s["quality"]), fontsize=7, alpha=0.7,
                    xytext=(4, 2), textcoords="offset points")
    o = out["refs"]["oracle"]
    ax.scatter(o["cost"] * 1000, o["quality"], color="black", marker="*", s=120, label="oracle", zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("mean cost, $ per 1k prompts (log)")
    ax.set_ylabel("mean quality (held-out)")
    ax.set_title("Cost–quality frontier: cluster baseline vs kNN router (RouterBench 0-shot)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "frontier.png"), dpi=150)


def report(out):
    refs = out["refs"]
    lines = [
        "# Router eval report",
        "",
        f"- Embedder: `{out['embedder']}` (shared by both routers)",
        f"- Train/test: {out['n_train']}/{out['n_test']} prompts (stratified by source, seed 42)",
        f"- Oracle: quality {refs['oracle']['quality']:.3f} @ ${refs['oracle']['cost']*1000:.2f}/1k",
        f"- Best single model: `{refs['best_single']}` "
        f"(quality {refs['singles'][refs['best_single']]['quality']:.3f} @ ${refs['singles'][refs['best_single']]['cost']*1000:.2f}/1k)",
        f"- Cheapest model: `{refs['cheapest']}` "
        f"(quality {refs['singles'][refs['cheapest']]['quality']:.3f} @ ${refs['singles'][refs['cheapest']]['cost']*1000:.2f}/1k)",
        "",
    ]
    budgets_usd_per_1k = [0.1, 0.25, 0.5, 1.0, 2.0, 3.5]
    header = "| router | " + " | ".join(f"q@${b}/1k" for b in budgets_usd_per_1k) + " | peak q | gap recovered @ peak |"
    lines += ["## Quality at matched cost", "", header, "|" + "---|" * (len(budgets_usd_per_1k) + 3)]
    for name, pts in out["routers"].items():
        points = [evals.Point(**p) for p in pts]
        cells = []
        for b in budgets_usd_per_1k:
            q = evals.quality_at_cost(points, b / 1000)
            cells.append("—" if q != q else f"{q:.3f}")
        best = max(points, key=lambda p: p.quality)
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {best.quality:.3f} | {best.gap_recovered:.1%} |")
    lines += ["", "![frontier](frontier.png)", ""]
    with open(os.path.join(RESULTS, "report.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
