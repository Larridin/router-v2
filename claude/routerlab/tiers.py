"""Tier-generalized routing (low/mid/high) vs model-specific routing.

Decomposes what the tier abstraction costs and buys:
  (a) 11-way ensemble (reference, from results/final.json)
  (b) 11-way estimates masked to 3 tier-representative models  -> action-space cost
  (c) heads trained on tier-aggregate targets, served on reps  -> generalization cost
  (d) model-specific heads @ 3k rows   (data-poor control)
  (e) tier-aggregate heads @ 3k rows   -> does the abstraction pay when data-scarce?

Tiers are cost bands over the RouterBench roster; the representative is the
tier's best-mean-quality member on the fit fold. Tier training target is
"can this tier handle it" (max quality over members); calibration is against
the REPRESENTATIVE's observed quality, so estimates stay honest about what is
actually served.

    python -m routerlab.tiers
"""

import json
import os
import time

import numpy as np

from . import dataset, evals
from .estimators import GBTEstimator
from .hypotheses import BUDGETS, load_folds, q_at
from .rules import ALPHAS, LAMBDAS, AlphaBlendRule, DollarEVRule, PromptCostModel

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")

TIERS = {  # cost bands over dataset.MODELS
    "low": ["mistralai/mistral-7b-chat", "WizardLM/WizardLM-13B-V1.2"],
    "mid": [
        "mistralai/mixtral-8x7b-chat", "meta/code-llama-instruct-34b-chat",
        "zero-one-ai/Yi-34B-Chat", "meta/llama-2-70b-chat",
        "claude-instant-v1", "gpt-3.5-turbo-1106",
    ],
    "high": ["claude-v1", "claude-v2", "gpt-4-1106-preview"],
}


def frontier(est3, rep_idx, alpha_rule, ev_rule, test, refs):
    """Sweep both rules over a (n,3) tier-estimate matrix; serve rep models."""
    pts = []
    for rule in (alpha_rule, ev_rule):
        for p in rule.params:
            ch3 = rule.choices(est3, test.texts, float(p))
            ch = rep_idx[ch3]  # tier -> served model column
            rq, rc = evals.realized(ch, test.quality, test.cost)
            lo, hi = evals.bootstrap_ci(rq)
            pts.append({"param": float(p), "rule": rule.name, "quality": float(rq.mean()),
                        "cost": float(rc.mean()), "q_lo": lo, "q_hi": hi})
    return pts


def main():
    t0 = time.time()
    train, test, test_emb, (fit_t, fit_e, fit_q, fit_c), (cal_t, cal_e, cal_q, cal_c) = load_folds()
    refs = evals.references(test.quality, test.cost, dataset.MODELS)

    cols = {name: [dataset.MODELS.index(m) for m in members] for name, members in TIERS.items()}
    rep_idx = np.array([
        cols[t][int(np.argmax(fit_q[:, cols[t]].mean(axis=0)))] for t in ("low", "mid", "high")
    ])
    reps = [dataset.MODELS[i] for i in rep_idx]
    print(f"tier representatives: low={reps[0]} mid={reps[1]} high={reps[2]}")

    # Tier-aggregate targets: max quality over tier members ("can this tier handle it").
    agg = lambda q: np.column_stack([q[:, cols[t]].max(axis=1) for t in ("low", "mid", "high")])
    rep = lambda q: q[:, rep_idx]

    alpha_rule = AlphaBlendRule(fit_c[:, rep_idx].mean(axis=0))
    ev_rule = DollarEVRule(PromptCostModel().fit(fit_t, fit_c[:, rep_idx]))
    out = {"reps": reps, "frontiers": {}}

    # (b) 11-way ensemble estimates masked to the 3 reps (needs the saved bundle).
    from .bundle import Router

    r = Router()
    print(f"loaded bundle {r.version} @ {time.time()-t0:.0f}s")
    p11 = r.ensemble.estimate_calibrated(test_emb, test.texts)
    out["frontiers"]["(b) model-heads full-data, 3-model roster"] = frontier(
        p11[:, rep_idx], rep_idx, alpha_rule, ev_rule, test, refs)
    print(f"(b) done @ {time.time()-t0:.0f}s")

    # (c) tier-aggregate heads, full data, calibrated to rep quality.
    tier_full = GBTEstimator()
    tier_full.fit(fit_e, agg(fit_q), fit_t)
    tier_full.calibrate(cal_e, rep(cal_q), cal_t)
    out["frontiers"]["(c) tier-heads full-data"] = frontier(
        tier_full.estimate_calibrated(test_emb, test.texts), rep_idx, alpha_rule, ev_rule, test, refs)
    print(f"(c) done @ {time.time()-t0:.0f}s")

    # (d)/(e): data-poor regime, same 3k rows for both.
    rng = np.random.default_rng(11)
    sub = np.sort(rng.choice(len(fit_t), size=3000, replace=False))
    sub_t = [fit_t[j] for j in sub]
    csub = np.sort(rng.choice(len(cal_t), size=600, replace=False))
    csub_t = [cal_t[j] for j in csub]

    model3k = GBTEstimator()
    model3k.fit(fit_e[sub], rep(fit_q[sub]), sub_t)
    model3k.calibrate(cal_e[csub], rep(cal_q[csub]), csub_t)
    out["frontiers"]["(d) model-heads @3k"] = frontier(
        model3k.estimate_calibrated(test_emb, test.texts), rep_idx, alpha_rule, ev_rule, test, refs)
    print(f"(d) done @ {time.time()-t0:.0f}s")

    tier3k = GBTEstimator()
    tier3k.fit(fit_e[sub], agg(fit_q[sub]), sub_t)
    tier3k.calibrate(cal_e[csub], rep(cal_q[csub]), csub_t)
    out["frontiers"]["(e) tier-heads @3k"] = frontier(
        tier3k.estimate_calibrated(test_emb, test.texts), rep_idx, alpha_rule, ev_rule, test, refs)
    print(f"(e) done @ {time.time()-t0:.0f}s")

    with open(os.path.join(RESULTS, "tiers.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Report: quality at matched cost (envelope over both rules per variant).
    ref11 = json.load(open(os.path.join(RESULTS, "final.json")))["frontiers"]
    env11 = ref11["ensembleFT × alpha"] + ref11["ensembleFT × dollar_ev"]
    lines = ["# Tier abstraction: what it costs, what it buys", "",
             f"- Tier reps: low=`{reps[0]}` mid=`{reps[1]}` high=`{reps[2]}`",
             f"- Reference oracle {refs['oracle']['quality']:.3f}; best single {refs['singles'][refs['best_single']]['quality']:.3f}", "",
             "| variant | " + " | ".join(f"q@${b}/1k" for b in BUDGETS) + " | peak |",
             "|" + "---|" * (len(BUDGETS) + 2)]
    rows = [("(a) model-heads full-data, 11-model roster", env11)] + list(out["frontiers"].items())
    for name, pts in rows:
        cells = " | ".join("—" if q_at(pts, b) != q_at(pts, b) else f"{q_at(pts, b):.3f}" for b in BUDGETS)
        peak = max(p["quality"] for p in pts)
        lines.append(f"| {name} | {cells} | {peak:.3f} |")
    with open(os.path.join(RESULTS, "tiers.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
