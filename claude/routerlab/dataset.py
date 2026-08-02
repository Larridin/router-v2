"""RouterBench loading: prompt corpus + per-(prompt, model) quality/cost matrices."""

import ast
from dataclasses import dataclass

import numpy as np
import pandas as pd

MODELS = [
    "WizardLM/WizardLM-13B-V1.2",
    "claude-instant-v1",
    "claude-v1",
    "claude-v2",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
    "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat",
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
]

MAX_PROMPT_CHARS = 1024  # mirror the router's TailTruncate cap


@dataclass
class Split:
    texts: list[str]  # tail-truncated prompt text, embedder input
    quality: np.ndarray  # (n, M) in [0, 1]
    cost: np.ndarray  # (n, M) realized $ per prompt per model
    source: np.ndarray  # (n,) eval_name, for stratification/breakdown


def prompt_text(raw) -> str:
    """RouterBench prompts are stringified lists of turns; join them."""
    s = str(raw)
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            s = "\n".join(str(x) for x in v)
    except (ValueError, SyntaxError):
        pass
    return s.strip()


def tail_truncate(s: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    return s if len(s) <= max_chars else s[-max_chars:]


def load(path: str) -> tuple[Split, Split]:
    """Load RouterBench, dedupe, stratified 70/30 prompt-level split (seed 42)."""
    df = pd.read_pickle(path)
    df = df.drop_duplicates(subset="prompt").reset_index(drop=True)

    texts = [tail_truncate(prompt_text(p)) for p in df["prompt"]]
    quality = df[MODELS].to_numpy(dtype=np.float64)
    cost = df[[m + "|total_cost" for m in MODELS]].to_numpy(dtype=np.float64)
    source = df["eval_name"].to_numpy()

    rng = np.random.default_rng(42)
    test_mask = np.zeros(len(df), dtype=bool)
    for src in np.unique(source):
        idx = np.flatnonzero(source == src)
        rng.shuffle(idx)
        test_mask[idx[: int(round(0.3 * len(idx)))]] = True

    def take(mask):
        return Split(
            texts=[t for t, m in zip(texts, mask) if m],
            quality=quality[mask],
            cost=cost[mask],
            source=source[mask],
        )

    return take(~test_mask), take(test_mask)
