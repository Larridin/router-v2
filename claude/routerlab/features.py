"""Hand-crafted text features to complement the sentence embedding (H1).

Cheap signals the 384-d embedding underweights: length (cost + difficulty
proxy), code/math markers, language, structure.
"""

import re

import numpy as np

_CODE = re.compile(r"```|\bdef\b|\bclass\b|\breturn\b|\bimport\b|[{};]")
_MATH = re.compile(r"[=+\-*/^<>]|\\frac|\\sum|\d+\.\d+")

FEATURE_NAMES = [
    "len_k", "log_len", "n_lines", "n_words", "avg_word_len",
    "frac_digits", "frac_nonascii", "frac_punct", "frac_upper",
    "code_hits_k", "math_hits_k", "is_question",
]


def text_features(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), len(FEATURE_NAMES)), dtype=np.float32)
    for i, t in enumerate(texts):
        n = max(len(t), 1)
        words = t.split()
        out[i] = [
            len(t) / 1000.0,
            np.log1p(len(t)),
            t.count("\n") + 1,
            len(words),
            (sum(len(w) for w in words) / max(len(words), 1)),
            sum(c.isdigit() for c in t) / n,
            sum(ord(c) > 127 for c in t) / n,
            sum(not c.isalnum() and not c.isspace() for c in t) / n,
            sum(c.isupper() for c in t) / n,
            len(_CODE.findall(t)) / n * 1000,
            len(_MATH.findall(t)) / n * 1000,
            1.0 if t.rstrip().endswith("?") else 0.0,
        ]
    return out
