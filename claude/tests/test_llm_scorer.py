import numpy as np

from routerlab.llm_scorer import DOMAINS, features, parse_score


def test_parse_clean_json():
    s, ok = parse_score('{"difficulty": 8, "domain": "math", "reasoning": true}')
    assert ok and s == {"difficulty": 8, "domain": "math", "reasoning": True}


def test_parse_json_embedded_in_prose():
    s, ok = parse_score('Sure! Here you go: {"difficulty": 3, "domain": "code", "reasoning": false} hope that helps')
    assert ok and s["difficulty"] == 3 and s["domain"] == "code"


def test_parse_garbage_falls_back():
    s, ok = parse_score("I cannot rate this prompt.")
    assert not ok and s == {"difficulty": 5, "domain": "other", "reasoning": False}


def test_parse_clamps_and_normalizes():
    s, ok = parse_score('{"difficulty": 99, "domain": "POETRY", "reasoning": 1}')
    assert ok and s["difficulty"] == 10 and s["domain"] == "other" and s["reasoning"] is True


def test_features_layout():
    F = features([{"difficulty": 10, "domain": "chinese", "reasoning": True}])
    assert F.shape == (1, 2 + len(DOMAINS))
    assert F[0, 0] == 1.0 and F[0, 1] == 1.0
    assert F[0, 2 + DOMAINS.index("chinese")] == 1.0 and F.sum() == 3.0
    assert np.isfinite(F).all()
