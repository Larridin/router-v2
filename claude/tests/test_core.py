import numpy as np
import pytest

from routerlab import core
from routerlab.evals import choose, realized, references


def test_zscore_per_prompt_hand_computed():
    q = np.array([[0.0, 1.0], [0.5, 0.5]])
    z = core.zscore_per_prompt(q)
    # row 0: mean .5, std .5 -> [-1, 1]; row 1: zero variance -> zeros
    assert z == pytest.approx(np.array([[-1.0, 1.0], [0.0, 0.0]]))


def test_shrink_pulls_small_groups_to_global():
    group = np.array([[1.0, 0.0]])
    global_mean = np.array([0.5, 0.5])
    # n=0 -> exactly global; n→inf -> exactly group mean
    assert core.shrink(group, np.array([0.0]), global_mean, 10.0)[0] == pytest.approx([0.5, 0.5])
    assert core.shrink(group, np.array([1e12]), global_mean, 10.0)[0] == pytest.approx([1.0, 0.0], abs=1e-9)
    # n=10, k0=10 -> midpoint
    assert core.shrink(group, np.array([10.0]), global_mean, 10.0)[0] == pytest.approx([0.75, 0.25])


def test_minmax_rows():
    x = np.array([[2.0, 4.0, 3.0], [1.0, 1.0, 1.0]])
    out = core.minmax_rows(x)
    assert out[0] == pytest.approx([0.0, 1.0, 0.5])
    assert out[1] == pytest.approx([0.0, 0.0, 0.0])  # constant row -> zeros


def test_blend_matches_manual_sum_over_clusters():
    rows = np.array([[0.0, 1.0], [1.0, 0.0]])  # two clusters, two models
    cost_norm = np.array([0.0, 1.0])  # model0 cheapest
    # alpha=1: pure quality, both rows minmax to themselves -> sums [1, 1]
    assert core.blend(rows, cost_norm, 1.0) == pytest.approx([1.0, 1.0])
    # alpha=0: pure cost over 2 rows -> [2·(1-0), 2·(1-1)] = [2, 0]
    assert core.blend(rows, cost_norm, 0.0) == pytest.approx([2.0, 0.0])


def test_argmax_stable_tie_breaks_first():
    assert core.argmax_stable(np.array([0.5, 0.5, 0.1])) == 0


def test_choose_alpha_extremes():
    # Q says model1 is better; cost says model0 is cheaper.
    Q = np.array([[0.2, 0.9]])
    cost_norm = np.array([0.0, 1.0])
    assert choose(Q, cost_norm, 1.0, alpha=1.0)[0] == 1
    assert choose(Q, cost_norm, 1.0, alpha=0.0)[0] == 0


def test_references_oracle_prefers_cheapest_tie():
    quality = np.array([[1.0, 1.0], [0.0, 1.0]])
    cost = np.array([[0.001, 0.01], [0.001, 0.01]])
    refs = references(quality, cost, ["cheap", "pricey"])
    # prompt0: tie -> cheap (cost .001); prompt1: pricey wins (cost .01)
    assert refs["oracle"]["quality"] == pytest.approx(1.0)
    assert refs["oracle"]["cost"] == pytest.approx((0.001 + 0.01) / 2)
    assert refs["cheapest"] == "cheap"
    assert refs["best_single"] == "pricey"


def test_realized_indexes_choice_columns():
    quality = np.array([[0.1, 0.9], [0.8, 0.2]])
    cost = np.array([[1.0, 2.0], [3.0, 4.0]])
    rq, rc = realized(np.array([1, 0]), quality, cost)
    assert rq == pytest.approx([0.9, 0.8])
    assert rc == pytest.approx([2.0, 3.0])
