"""Load LLMRouterBench data into the Split format expected by routerlab estimators.

Points at the router-bench project's downloaded data by default.
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Default: sibling router-bench project's downloaded data
_DEFAULT_BENCH = os.path.join(
    os.path.dirname(__file__), "..", "..", "router-bench", "results", "bench"
)

# Flagship model roster from LLMRouterBench performance-cost setting
LLM_MODELS = [
    "claude-sonnet-4",
    "deepseek-v3-0324",
    "deepseek-v3.1-terminus",
    "deepseek-r1-0528",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gpt-5-chat",
    "gpt-5",
    "qwen3-235b-a22b-2507",
    "qwen3-235b-a22b-thinking-2507",
    "glm-4.6",
    "kimi-k2-0905",
    "intern-s1",
]

# Coding-relevant datasets from LLMRouterBench
LLM_DATASETS = [
    "aime", "livemathbench", "gpqa", "hle",
    "livecodebench", "mmlupro", "swe-bench",
]

MAX_PROMPT_CHARS = 1024


@dataclass
class Split:
    texts: list[str]
    quality: np.ndarray   # (n, M) in [0, 1]
    cost: np.ndarray      # (n, M) in USD
    source: np.ndarray    # (n,) dataset name


def tail_truncate(s: str, max_chars: int = MAX_PROMPT_CHARS) -> str:
    return s if len(s) <= max_chars else s[-max_chars:]


def _load_model_file(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)["records"]


def load(bench_dir: str = _DEFAULT_BENCH) -> tuple[Split, Split]:
    """Load LLMRouterBench, pivot to per-prompt quality/cost matrices, stratified split.

    Returns (train, test) with 80/20 prompt-level split per dataset.
    """
    bench = Path(bench_dir)
    prompts: dict[tuple, dict] = {}  # (dataset, prompt_text) -> {model: score/cost}

    for dataset in LLM_DATASETS:
        ds_dir = bench / dataset
        if not ds_dir.is_dir():
            continue
        for split_dir in ds_dir.iterdir():
            if not split_dir.is_dir():
                continue
            for model_dir in split_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model_name = model_dir.name.lower().replace("_", "-").replace(" ", "-")
                # Map some name variations
                name_map = {
                    "claude-4-sonnet": "claude-sonnet-4",
                    "gpt5": "gpt-5",
                }
                model_name = name_map.get(model_name, model_name)

                if model_name not in LLM_MODELS:
                    continue

                for fpath in model_dir.glob("*.json"):
                    try:
                        records = _load_model_file(str(fpath))
                    except (json.JSONDecodeError, KeyError):
                        continue
                    for rec in records:
                        prompt = rec.get("prompt", "")
                        if not prompt:
                            continue
                        key = (dataset, tail_truncate(prompt))
                        if key not in prompts:
                            prompts[key] = {"dataset": dataset, "prompt": prompt, "records": {}, "usages": {}}
                        prompts[key]["records"][model_name] = float(rec.get("score", 0) or 0)
                        prompts[key]["usages"][model_name] = {
                            "cost": float(rec.get("cost", 0) or 0),
                            "prompt_tokens": int(rec.get("prompt_tokens", 0) or 0),
                            "completion_tokens": int(rec.get("completion_tokens", 0) or 0),
                        }

    if not prompts:
        raise RuntimeError(f"No LLMRouterBench data found at {bench_dir}. "
                           f"Download from: https://huggingface.co/datasets/NPULH/LLMRouterBench")

    items = list(prompts.values())
    texts = [tail_truncate(item["prompt"]) for item in items]

    # Build quality and cost matrices
    quality = np.zeros((len(items), len(LLM_MODELS)))
    cost = np.zeros((len(items), len(LLM_MODELS)))
    source = np.array([item["dataset"] for item in items])

    for i, item in enumerate(items):
        for j, model in enumerate(LLM_MODELS):
            quality[i, j] = float(item["records"].get(model, 0) or 0)
            cost[i, j] = float(item["usages"].get(model, {}).get("cost", 0) or 0)

    # Stratified 80/20 split by dataset
    rng = np.random.default_rng(42)
    test_mask = np.zeros(len(items), dtype=bool)
    for ds in np.unique(source):
        idx = np.flatnonzero(source == ds)
        rng.shuffle(idx)
        n_test = int(round(0.2 * len(idx)))
        test_mask[idx[:n_test]] = True

    def take(mask):
        return Split(
            texts=[t for t, m in zip(texts, mask) if m],
            quality=quality[mask],
            cost=cost[mask],
            source=source[mask],
        )

    train = take(~test_mask)
    test = take(test_mask)
    print(f"LLMRouterBench: {len(train.texts)} train, {len(test.texts)} test prompts "
          f"from {len(np.unique(source))} datasets, {len(LLM_MODELS)} models")
    return train, test