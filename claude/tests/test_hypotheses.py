import numpy as np
import pytest

from routerlab.features import FEATURE_NAMES, text_features
from routerlab.rules import AlphaBlendRule, DollarEVRule, PromptCostModel


def test_text_features_shape_and_finite():
    X = text_features(["hello world?", "def f(x):\n    return x + 1", "古诗词" * 50, ""])
    assert X.shape == (4, len(FEATURE_NAMES))
    assert np.isfinite(X).all()
    assert X[0, FEATURE_NAMES.index("is_question")] == 1.0
    assert X[1, FEATURE_NAMES.index("code_hits_k")] > 0
    assert X[2, FEATURE_NAMES.index("frac_nonascii")] > 0.9


def test_prompt_cost_model_recovers_linear():
    texts = ["x" * n for n in [10, 100, 1000, 5000]]
    cost = np.column_stack([
        [1e-4 + 2e-7 * len(t) for t in texts],   # model A: a=1e-4, b=2e-7
        [5e-5 + 0.0 * len(t) for t in texts],    # model B: flat
    ])
    cm = PromptCostModel().fit(texts, cost)
    pred = cm.predict(["y" * 2000])
    assert pred[0, 0] == pytest.approx(1e-4 + 2e-7 * 2000, rel=1e-6)
    assert pred[0, 1] == pytest.approx(5e-5, rel=1e-6)


def test_dollar_ev_extremes():
    texts = ["aaaa", "bbbb"]
    cost = np.array([[0.001, 0.00001], [0.001, 0.00001]])
    rule = DollarEVRule(PromptCostModel().fit(texts, cost))
    est = np.array([[0.9, 0.5], [0.9, 0.5]])  # model0 better, model1 far cheaper
    assert (rule.choices(est, texts, 0.0) == 0).all()  # lambda=0: pure quality
    assert (rule.choices(est, texts, 1e6) == 1).all()  # huge lambda: pure cost


def test_alpha_rule_extremes():
    rule = AlphaBlendRule(np.array([0.001, 0.01]))
    est = np.array([[0.1, 0.9]])
    assert rule.choices(est, ["t"], 1.0)[0] == 1
    assert rule.choices(est, ["t"], 0.0)[0] == 0
