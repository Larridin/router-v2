"""Decision rules: estimate matrix -> chosen model, swept over a tradeoff knob.

AlphaBlendRule — the incumbent: per-prompt min-max quality vs a normalized
mean-cost axis, alpha in [0,1].

DollarEVRule (H2) — calibrated expected utility in raw dollars:
    argmax_m  P̂(success)[m] − lambda · ĉ(prompt, m)
with a per-prompt cost prediction ĉ fit per model as cost ≈ a + b·len(prompt)
on the fit fold (input cost scales with prompt length; a absorbs the model's
typical output spend). lambda sweeps the frontier in $-per-quality units.
"""

import numpy as np

from . import core

ALPHAS = np.round(np.linspace(0.0, 1.0, 21), 3)
LAMBDAS = np.concatenate([[0.0], np.geomspace(1.0, 3e5, 27)])


class PromptCostModel:
    """Per-model linear cost-in-prompt-length: ĉ[m] = max(a_m + b_m·len, floor_m).

    floor_m is the model's 10th-percentile observed cost: a model never costs
    less than its typical output spend, and without the floor an OLS intercept
    predicts ~zero for short prompts — flattening the cost signal exactly where
    the EV rule needs it (all models look free, everything routes to the
    quality argmax).
    """

    def fit(self, texts: list[str], cost: np.ndarray):
        L = np.array([len(t) for t in texts], dtype=np.float64)
        self.coef = []
        self.floor = np.quantile(cost, 0.10, axis=0)
        for m in range(cost.shape[1]):
            b, a = np.polyfit(L, cost[:, m], 1)
            self.coef.append((a, b))
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        L = np.array([len(t) for t in texts], dtype=np.float64)
        out = np.column_stack([a + b * L for a, b in self.coef])
        return np.clip(out, np.clip(self.floor, 1e-7, None)[None, :], None)


class AlphaBlendRule:
    name = "alpha"
    params = ALPHAS

    def __init__(self, mean_cost_per_model: np.ndarray):
        self.cost_norm = core.cost_axis(mean_cost_per_model)

    def choices(self, est: np.ndarray, texts: list[str], param: float) -> np.ndarray:
        q = core.minmax_rows(est)
        return np.argmax(param * q + (1.0 - param) * (1.0 - self.cost_norm[None, :]), axis=1)


class DollarEVRule:
    name = "dollar_ev"
    params = LAMBDAS

    def __init__(self, cost_model: PromptCostModel):
        self.cost_model = cost_model
        self._cache_key = None

    def _cost_hat(self, texts: list[str]) -> np.ndarray:
        key = id(texts)
        if self._cache_key != key:
            self._cache = self.cost_model.predict(texts)
            self._cache_key = key
        return self._cache

    def choices(self, est: np.ndarray, texts: list[str], param: float) -> np.ndarray:
        utility = est - param * self._cost_hat(texts)
        return np.argmax(utility, axis=1)
