"""Router evals: route each labeled prompt, look up what the pick earned.

A router is (Q, cost_norm, cost_scale) where Q is the precomputed (n, M)
effective-quality matrix. For each alpha the choice is
    argmax_m alpha·Q[m] + (1−alpha)·cost_scale·(1−cost_norm[m]).

Metrics per (router, alpha): mean realized quality, mean realized $ per prompt,
oracle-gap-recovered, bootstrap CIs. Reference points: oracle (per-prompt best
model, cheapest among ties), best single model, cheapest model, uniform random.
"""

from dataclasses import dataclass

import numpy as np

ALPHAS = np.round(np.linspace(0.0, 1.0, 21), 3)
BOOTSTRAP = 1000


@dataclass
class Point:
    alpha: float
    quality: float
    cost: float
    q_lo: float
    q_hi: float
    gap_recovered: float
    mix: dict  # model -> share of prompts


def choose(Q: np.ndarray, cost_norm: np.ndarray, cost_scale: float, alpha: float) -> np.ndarray:
    scores = alpha * Q + (1.0 - alpha) * cost_scale * (1.0 - cost_norm[None, :])
    return np.argmax(scores, axis=1)


def realized(choices: np.ndarray, quality: np.ndarray, cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(len(choices))
    return quality[rows, choices], cost[rows, choices]


def bootstrap_ci(values: np.ndarray, n: int = BOOTSTRAP, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def references(quality: np.ndarray, cost: np.ndarray, models: list[str]) -> dict:
    """Oracle + single-model reference points on the same prompts."""
    # Oracle: max quality per prompt; among ties pick the cheapest model.
    best_q = quality.max(axis=1, keepdims=True)
    tie_cost = np.where(quality >= best_q, cost, np.inf)
    oracle_idx = np.argmin(tie_cost, axis=1)
    oq, oc = realized(oracle_idx, quality, cost)

    singles = {
        m: {"quality": float(quality[:, j].mean()), "cost": float(cost[:, j].mean())}
        for j, m in enumerate(models)
    }
    best_single = max(singles, key=lambda m: singles[m]["quality"])
    cheapest = min(singles, key=lambda m: singles[m]["cost"])
    return {
        "oracle": {"quality": float(oq.mean()), "cost": float(oc.mean())},
        "singles": singles,
        "best_single": best_single,
        "cheapest": cheapest,
        "random": {
            "quality": float(quality.mean()),
            "cost": float(cost.mean()),
        },
    }


def sweep(router, Q: np.ndarray, quality: np.ndarray, cost: np.ndarray, models: list[str], refs: dict) -> list[Point]:
    o_q = refs["oracle"]["quality"]
    c_q = refs["singles"][refs["cheapest"]]["quality"]
    points = []
    for a in ALPHAS:
        ch = choose(Q, router.cost_norm, router.cost_scale, float(a))
        rq, rc = realized(ch, quality, cost)
        lo, hi = bootstrap_ci(rq)
        counts = np.bincount(ch, minlength=len(models)) / len(ch)
        gap = (rq.mean() - c_q) / (o_q - c_q) if o_q > c_q else 0.0
        points.append(
            Point(
                alpha=float(a),
                quality=float(rq.mean()),
                cost=float(rc.mean()),
                q_lo=lo,
                q_hi=hi,
                gap_recovered=float(gap),
                mix={m: float(counts[j]) for j, m in enumerate(models) if counts[j] > 0},
            )
        )
    return points


def quality_at_cost(points: list[Point], budget: float) -> float:
    """Best quality achievable on the router's frontier at mean cost <= budget."""
    ok = [p.quality for p in points if p.cost <= budget]
    return max(ok) if ok else float("nan")
