"""Pure scoring math shared by both routers.

Mirrors the production router's trainer/scorer semantics:
- per-prompt z-score across models (metadata: per_prompt_zscore_across_bench_columns)
- empirical-Bayes shrinkage toward the model's global mean (shrinkage_k0)
- per-cluster min-max quality normalization + alpha blend with a globally
  min-max-normalized cost axis (cluster/scorer.go blendScoresV2)
"""

import numpy as np


def zscore_per_prompt(quality: np.ndarray) -> np.ndarray:
    """Z-score each row (prompt) across models. Zero-variance rows -> zeros."""
    mu = quality.mean(axis=1, keepdims=True)
    sd = quality.std(axis=1, keepdims=True)
    out = np.zeros_like(quality, dtype=np.float64)
    np.divide(quality - mu, sd, out=out, where=sd > 0)
    return out


def shrink(group_mean: np.ndarray, group_n: np.ndarray, global_mean: np.ndarray, k0: float) -> np.ndarray:
    """(n·mean + k0·global) / (n + k0), broadcasting n over model columns."""
    n = np.asarray(group_n, dtype=np.float64)[:, None]
    return (n * group_mean + k0 * global_mean[None, :]) / (n + k0)


def minmax_rows(x: np.ndarray) -> np.ndarray:
    """Min-max normalize each row; constant rows -> zeros."""
    lo = x.min(axis=1, keepdims=True)
    rng = x.max(axis=1, keepdims=True) - lo
    out = np.zeros_like(x, dtype=np.float64)
    np.divide(x - lo, rng, out=out, where=rng > 0)
    return out


def blend(quality_rows: np.ndarray, cost_norm: np.ndarray, alpha: float) -> np.ndarray:
    """Sum alpha·qNorm + (1−alpha)·(1−cNorm) over rows (top-p clusters or one row).

    quality_rows: (p, M) UN-normalized quality rows; each row is min-max
    normalized independently (per-cluster normalization, as in blendScoresV2).
    cost_norm: (M,) globally min-max-normalized cost axis.
    Returns (M,) scores.
    """
    q = minmax_rows(np.atleast_2d(quality_rows))
    per_row = alpha * q + (1.0 - alpha) * (1.0 - cost_norm[None, :])
    return per_row.sum(axis=0)


def argmax_stable(scores: np.ndarray) -> int:
    """First index of the max (ties broken by model order, like the Go argmax)."""
    return int(np.argmax(scores))


def cost_axis(mean_cost_per_model: np.ndarray) -> np.ndarray:
    """Globally min-max-normalized model cost axis."""
    lo, hi = mean_cost_per_model.min(), mean_cost_per_model.max()
    if hi - lo <= 0:
        return np.zeros_like(mean_cost_per_model)
    return (mean_cost_per_model - lo) / (hi - lo)
