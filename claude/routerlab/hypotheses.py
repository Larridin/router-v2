"""H1/H2/H3 vs the champion. Usage:

    python -m routerlab.hypotheses h1        # GBT estimator × both rules
    python -m routerlab.hypotheses h2        # classic estimators × dollar-EV rule (+ champion)
    python -m routerlab.hypotheses h3        # calibrated ensemble × both rules
    python -m routerlab.hypotheses combine   # leaderboard + plot + verdict

Folds: train (25.5k) -> fit 80% / calibration 20% (seed 7); test untouched.
The champion (cluster k=256, alpha rule) keeps its FULL-train fit — the
hypotheses run with a data handicap, so any win is real.
"""

import json
import os
import sys
import time

import numpy as np

from . import dataset, evals
from .baseline import ClusterRouter
from .embedder import embed_cached, get_embedder
from .estimators import ClusterEstimator, EnsembleEstimator, GBTEstimator, KNNEstimator
from .rules import AlphaBlendRule, DollarEVRule, PromptCostModel

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
BUDGETS = [0.25, 0.5, 1.0, 2.0, 3.5]  # $ per 1k prompts


def load_folds():
    train, test = dataset.load(os.path.join(DATA, "routerbench_0shot.pkl"))
    emb = get_embedder()
    train_emb = embed_cached(emb, train.texts, DATA)
    test_emb = embed_cached(emb, test.texts, DATA)
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(train.texts))
    n_fit = int(0.8 * len(idx))
    fit_i, cal_i = np.sort(idx[:n_fit]), np.sort(idx[n_fit:])
    fold = lambda i: (
        [train.texts[j] for j in i], train_emb[i], train.quality[i], train.cost[i]
    )
    return train, test, test_emb, fold(fit_i), fold(cal_i)


def frontier(est_matrix, rule, texts, quality, cost, refs) -> list[dict]:
    pts = []
    o_q = refs["oracle"]["quality"]
    c_q = refs["singles"][refs["cheapest"]]["quality"]
    for p in rule.params:
        ch = rule.choices(est_matrix, texts, float(p))
        rq, rc = evals.realized(ch, quality, cost)
        lo, hi = evals.bootstrap_ci(rq)
        pts.append({
            "param": float(p), "quality": float(rq.mean()), "cost": float(rc.mean()),
            "q_lo": lo, "q_hi": hi,
            "gap_recovered": float((rq.mean() - c_q) / (o_q - c_q)) if o_q > c_q else 0.0,
        })
    return pts


def run(which: str):
    t0 = time.time()
    train, test, test_emb, (fit_t, fit_e, fit_q, fit_c), (cal_t, cal_e, cal_q, cal_c) = load_folds()
    refs = evals.references(test.quality, test.cost, dataset.MODELS)
    alpha_rule = AlphaBlendRule(fit_c.mean(axis=0))
    ev_rule = DollarEVRule(PromptCostModel().fit(fit_t, fit_c))
    out = {}

    def add(tag, est_matrix):
        for rule in (alpha_rule, ev_rule):
            out[f"{tag} × {rule.name}"] = frontier(est_matrix, rule, test.texts, test.quality, test.cost, refs)
            print(f"{which}: {tag} × {rule.name} done @ {time.time()-t0:.0f}s")

    if which == "h1":
        gbt = GBTEstimator().fit(fit_e, fit_q, fit_t).calibrate(cal_e, cal_q, cal_t)
        add("gbt", gbt.estimate_calibrated(test_emb, test.texts))
    elif which == "h2":
        # Champion: full-train fit, alpha rule (published benchmark), plus EV variants.
        champ = ClusterRouter(k=256, n_init=3).fit(
            np.concatenate([fit_e, cal_e]), np.concatenate([fit_q, cal_q]), np.concatenate([fit_c, cal_c])
        )
        out["champion cluster256 × alpha"] = frontier(
            champ.precompute(test_emb), AlphaBlendRule(np.concatenate([fit_c, cal_c]).mean(axis=0)),
            test.texts, test.quality, test.cost, refs,
        )
        print(f"h2: champion done @ {time.time()-t0:.0f}s")
        for est in (ClusterEstimator(), KNNEstimator()):
            est.fit(fit_e, fit_q, fit_t).calibrate(cal_e, cal_q, cal_t)
            add(est.name, est.estimate_calibrated(test_emb, test.texts))
    elif which == "h3":
        members = [ClusterEstimator(), KNNEstimator(), GBTEstimator()]
        for m in members:
            m.fit(fit_e, fit_q, fit_t).calibrate(cal_e, cal_q, cal_t)
            print(f"h3: fitted {m.name} @ {time.time()-t0:.0f}s")
        add("ensemble", EnsembleEstimator(members).estimate_calibrated(test_emb, test.texts))
    elif which == "final":
        # Ensemble without the data handicap: calibrators come from the 80/20
        # protocol (fit on 80%, isotonic on 20%), then members are REFIT on the
        # full train split. Calibrator/estimator distribution mismatch is the
        # standard tradeoff; monotone maps tolerate it.
        full_t = fit_t + cal_t
        full_e = np.concatenate([fit_e, cal_e])
        full_q = np.concatenate([fit_q, cal_q])
        members = [ClusterEstimator(), KNNEstimator(), GBTEstimator()]
        for m in members:
            m.fit(fit_e, fit_q, fit_t).calibrate(cal_e, cal_q, cal_t)
            m.fit(full_e, full_q, full_t)
            print(f"final: fitted {m.name} on full train @ {time.time()-t0:.0f}s")
        add("ensembleFT", EnsembleEstimator(members).estimate_calibrated(test_emb, test.texts))
    elif which in ("h4", "h4pilot"):
        from .estimators import _Calibrated
        from .llm_scorer import Scorer, features, latency_stats

        rng = np.random.default_rng(11)
        n_sub, n_cal = (3000, 600) if which == "h4" else (200, 0)
        sub_i = np.sort(rng.choice(len(fit_t), size=n_sub, replace=False))
        scorer = Scorer()
        sub_texts = [fit_t[j] for j in sub_i]
        rows_fit = scorer.score(sub_texts)
        print(f"{which}: scored {len(rows_fit)} fit prompts @ {time.time()-t0:.0f}s; {latency_stats(rows_fit)}")

        if which == "h4pilot":
            F = features(rows_fit)
            mean_q = fit_q[sub_i].mean(axis=1)  # per-prompt mean quality across models
            hard = F[:, 0]
            corr = float(np.corrcoef(hard, mean_q)[0, 1])
            print(f"pilot: difficulty vs mean-quality corr = {corr:+.3f} (want strongly negative)")
            print(f"pilot: difficulty histogram {np.histogram(hard*10, bins=np.arange(12))[0].tolist()}")
            est_cost = (len(fit_t) and (3600 + len(test.texts)) * 0.0006)
            print(f"pilot: full-run label estimate ≈ ${est_cost:.2f}")
            return

        cal_i = np.sort(rng.choice(len(cal_t), size=n_cal, replace=False))
        cal_texts_sub = [cal_t[j] for j in cal_i]
        rows_cal = scorer.score(cal_texts_sub)
        rows_test = scorer.score(test.texts)
        lat = latency_stats(rows_fit + rows_cal + rows_test)
        print(f"h4: all labeled @ {time.time()-t0:.0f}s; latency {lat}")

        from sklearn.ensemble import HistGradientBoostingRegressor

        class _FeatEst(_Calibrated):
            """GBT head over an arbitrary precomputed feature matrix."""

            def __init__(self, name):
                self.name = name

            def fit_mat(self, X, quality):
                self.models = []
                for m in range(quality.shape[1]):
                    g = HistGradientBoostingRegressor(
                        max_iter=300, learning_rate=0.08, max_leaf_nodes=63,
                        early_stopping=True, validation_fraction=0.1, random_state=42,
                    )
                    g.fit(X, quality[:, m])
                    self.models.append(g)
                return self

            def est_mat(self, X):
                return np.clip(np.column_stack([g.predict(X) for g in self.models]), 0.0, 1.0)

            # _Calibrated adapters: we bypass estimate(emb, texts) and work on matrices.
            def calibrate_mat(self, X, quality):
                raw = self.est_mat(X)
                from sklearn.isotonic import IsotonicRegression

                self._iso = []
                for m in range(quality.shape[1]):
                    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                    iso.fit(raw[:, m], quality[:, m])
                    self._iso.append(iso)
                return self

            def est_cal_mat(self, X):
                raw = self.est_mat(X)
                return np.column_stack([iso.predict(raw[:, m]) for m, iso in enumerate(self._iso)])

        F_fit, F_cal, F_test = features(rows_fit), features(rows_cal), features(rows_test)
        E_fit, E_cal, E_test = fit_e[sub_i], cal_e[cal_i], test_emb
        variants = {
            "llmfeat3k": (F_fit, F_cal, F_test),
            "emb3k": (E_fit, E_cal, E_test),  # control: same 3k rows, embedding only
            "hybrid3k": (
                np.hstack([E_fit, F_fit]), np.hstack([E_cal, F_cal]), np.hstack([E_test, F_test]),
            ),
        }
        for tag, (Xf, Xc, Xt) in variants.items():
            est = _FeatEst(tag).fit_mat(Xf, fit_q[sub_i]).calibrate_mat(Xc, cal_q[cal_i])
            add(tag, est.est_cal_mat(Xt))
        out["_latency"] = lat
    else:
        raise SystemExit(f"unknown mode {which}")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"{which}.json"), "w") as f:
        json.dump({"refs": refs, "frontiers": out}, f, indent=2)
    print(f"{which} done in {time.time()-t0:.0f}s")


def q_at(pts, budget_per_1k):
    ok = [p["quality"] for p in pts if p["cost"] * 1000 <= budget_per_1k]
    return max(ok) if ok else float("nan")


def combine():
    fronts, refs = {}, None
    for w in ("h1", "h2", "h3", "final", "h4"):
        path = os.path.join(RESULTS, f"{w}.json")
        if not os.path.exists(path):
            print(f"missing {path}; run `python -m routerlab.hypotheses {w}` first")
            continue
        blob = json.load(open(path))
        refs = blob["refs"]
        fronts.update({k: v for k, v in blob["frontiers"].items() if not k.startswith("_")})
        if "_latency" in blob["frontiers"]:
            lat = blob["frontiers"]["_latency"]
            print(f"[{w}] LLM routing-call latency: p50={lat.get('p50_ms', 0):.0f}ms "
                  f"p95={lat.get('p95_ms', 0):.0f}ms parse_fail={lat.get('parse_fail_rate', 0):.1%}")

    # Rule envelope per estimator: an operator picks the better rule per
    # budget, so the deployable frontier is the union of both rules' points.
    tags = {k.split(" × ")[0] for k in fronts if not k.startswith("champion")}
    for tag in sorted(tags):
        pair = [fronts[k] for k in (f"{tag} × alpha", f"{tag} × dollar_ev") if k in fronts]
        if len(pair) == 2:
            fronts[f"{tag} × envelope"] = pair[0] + pair[1]

    champ_name = "champion cluster256 × alpha"
    champ = fronts.get(champ_name)
    lines = ["# Hypothesis results vs champion", ""]
    lines += [f"- Oracle {refs['oracle']['quality']:.3f} @ ${refs['oracle']['cost']*1000:.2f}/1k; "
              f"best single `{refs['best_single']}` {refs['singles'][refs['best_single']]['quality']:.3f}", ""]
    hdr = "| frontier | " + " | ".join(f"q@${b}/1k" for b in BUDGETS) + " | peak q | peak $/1k |"
    lines += [hdr, "|" + "---|" * (len(BUDGETS) + 3)]
    order = [champ_name] + sorted(k for k in fronts if k != champ_name)
    for name in order:
        pts = fronts[name]
        best = max(pts, key=lambda p: p["quality"])
        cells = " | ".join("—" if q_at(pts, b) != q_at(pts, b) else f"{q_at(pts, b):.3f}" for b in BUDGETS)
        lines.append(f"| {name} | {cells} | {best['quality']:.3f} | {best['cost']*1000:.2f} |")

    if champ:
        lines += ["", "## Verdict vs champion (quality delta at matched cost)", ""]
        lines += ["| frontier | " + " | ".join(f"Δ@${b}" for b in BUDGETS) + " | wins |", "|" + "---|" * (len(BUDGETS) + 2)]
        for name in order[1:]:
            deltas = [q_at(fronts[name], b) - q_at(champ, b) for b in BUDGETS]
            wins = sum(d > 0.008 for d in deltas)  # outside bootstrap CI half-width
            lines.append(f"| {name} | " + " | ".join(f"{d:+.3f}" for d in deltas) + f" | {wins}/{len(BUDGETS)} |")

    with open(os.path.join(RESULTS, "hypotheses.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    plot(fronts, refs)
    print("\n".join(lines))


def plot(fronts, refs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for name, pts in sorted(fronts.items()):
        if name.endswith("envelope"):  # point unions zigzag when drawn as a line
            continue
        pts = sorted(pts, key=lambda p: p["cost"])
        style = dict(lw=2.5, color="black", zorder=5) if name.startswith("champion") else dict(lw=1.4, alpha=0.9)
        ax.plot([p["cost"] * 1000 for p in pts], [p["quality"] for p in pts], marker="o", ms=2.5, label=name, **style)
    for m, s in refs["singles"].items():
        ax.scatter(s["cost"] * 1000, s["quality"], color="gray", s=14, zorder=3)
    o = refs["oracle"]
    ax.scatter(o["cost"] * 1000, o["quality"], color="red", marker="*", s=140, label="oracle", zorder=6)
    ax.set_xscale("log")
    ax.set_xlabel("mean cost, $ per 1k prompts (log)")
    ax.set_ylabel("mean quality (held-out)")
    ax.set_title("Hypotheses vs champion — RouterBench 0-shot")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "hypotheses_frontier.png"), dpi=150)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "combine"
    combine() if mode == "combine" else run(mode)
