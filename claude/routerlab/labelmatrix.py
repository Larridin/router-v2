"""Build a label matrix for a current model roster via OpenRouter.

    python -m routerlab.labelmatrix corpus            # build data/labelcorpus.jsonl
    python -m routerlab.labelmatrix pilot             # haiku x 40 prompts, cost projection
    python -m routerlab.labelmatrix run [slug ...]    # full matrix (resumable, cached)
    python -m routerlab.labelmatrix report            # grade caches -> matrix + summary

Design: calling and grading are SEPARATE passes over a raw-response cache
(data/matrix/<slug>.jsonl), so a grader fix never re-spends API money, and
adding a model later = one more `run` invocation (incremental column).
Recorded per call: response text, prompt/completion tokens, OpenRouter-billed
cost, TTFT (streaming) and total latency.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
MATRIX_DIR = os.path.join(DATA, "matrix")
CORPUS = os.path.join(DATA, "labelcorpus.jsonl")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

ROSTER = [
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-pro",
    "moonshotai/kimi-k2.7-code",
    "anthropic/claude-fable-5",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-opus-5",
    "anthropic/claude-sonnet-5",
]

MAX_TOKENS = 2500
CONCURRENCY = 24

SOURCES = {"mmlu_pro": 500, "gsm8k": 400, "mbpp": 400}
URLS = {
    "mmlu_pro": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet",
    "gsm8k": "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet",
    "mbpp": "https://huggingface.co/datasets/google-research-datasets/mbpp/resolve/main/full/test-00000-of-00001.parquet",
}


# ---------- corpus ----------

def build_corpus():
    rng = np.random.default_rng(42)
    rows = []

    df = pd.read_parquet(URLS["mmlu_pro"])
    per_cat = max(1, SOURCES["mmlu_pro"] // df["category"].nunique())
    for cat, g in df.groupby("category"):
        take = g.iloc[rng.permutation(len(g))[: per_cat]]
        for _, r in take.iterrows():
            letters = [chr(ord("A") + i) for i in range(len(r["options"]))]
            opts = "\n".join(f"{a}. {o}" for a, o in zip(letters, r["options"]))
            rows.append({
                "id": f"mmlu_pro:{r['question_id']}", "source": "mmlu_pro",
                "prompt": f"{r['question']}\n\n{opts}\n\nThink briefly, then finish with exactly: Answer: <letter>",
                "gold": r["answer"],
            })

    df = pd.read_parquet(URLS["gsm8k"])
    take = df.iloc[rng.permutation(len(df))[: SOURCES["gsm8k"]]]
    for i, r in take.iterrows():
        gold = r["answer"].split("####")[-1].strip()
        rows.append({
            "id": f"gsm8k:{i}", "source": "gsm8k",
            "prompt": f"{r['question']}\n\nSolve step by step, then finish with exactly: Answer: <number>",
            "gold": gold,
        })

    df = pd.read_parquet(URLS["mbpp"])
    take = df.iloc[rng.permutation(len(df))[: SOURCES["mbpp"]]]
    for _, r in take.iterrows():
        tests = "\n".join(list(r["test_list"]))
        rows.append({
            "id": f"mbpp:{r['task_id']}", "source": "mbpp",
            "prompt": f"{r['text']}\n\nYour solution must pass these tests:\n{tests}\n\nReply with ONLY a Python code block.",
            "gold": json.dumps({"tests": list(r["test_list"]), "setup": r["test_setup_code"] or ""}),
        })

    os.makedirs(DATA, exist_ok=True)
    with open(CORPUS, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"corpus: {len(rows)} prompts -> {CORPUS} "
          f"({pd.Series([r['source'] for r in rows]).value_counts().to_dict()})")


def load_corpus() -> list[dict]:
    return [json.loads(l) for l in open(CORPUS)]


# ---------- graders (pure; unit-tested) ----------

_ANS_RE = re.compile(r"answer\s*[:\-]?\s*\**\s*([A-J]|-?[\d,]+(?:\.\d+)?)", re.IGNORECASE)


def extract_answer(text: str) -> str:
    hits = _ANS_RE.findall(text or "")
    return hits[-1].strip().upper().replace(",", "") if hits else ""


def grade_mcq(text: str, gold: str) -> float:
    return 1.0 if extract_answer(text) == gold.strip().upper() else 0.0


def grade_number(text: str, gold: str) -> float:
    got = extract_answer(text)
    try:
        return 1.0 if abs(float(got) - float(gold.replace(",", ""))) < 1e-6 else 0.0
    except ValueError:
        return 0.0


_CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    blocks = _CODE_RE.findall(text or "")
    return blocks[-1] if blocks else (text or "")


def grade_mbpp(text: str, gold: str, timeout: float = 10.0) -> float:
    payload = json.loads(gold)
    program = extract_code(text) + "\n\n" + payload["setup"] + "\n" + "\n".join(payload["tests"]) + "\nprint('__PASS__')\n"
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sol.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            out = subprocess.run([sys.executable, "-I", path], capture_output=True,
                                 timeout=timeout, cwd=td, text=True)
            return 1.0 if "__PASS__" in out.stdout else 0.0
        except (subprocess.TimeoutExpired, OSError):
            return 0.0


GRADERS = {"mmlu_pro": grade_mcq, "gsm8k": grade_number, "mbpp": grade_mbpp}


# ---------- runner ----------

def _key() -> str:
    return open(os.path.join(ROOT, ".openrouter_key")).read().strip()


def cache_path(slug: str) -> str:
    return os.path.join(MATRIX_DIR, slug.replace("/", "_") + ".jsonl")


def load_cache(slug: str) -> dict:
    path = cache_path(slug)
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                out[row["id"]] = row
    return out


def run_model(slug: str, prompts: list[dict], concurrency: int = CONCURRENCY):
    os.makedirs(MATRIX_DIR, exist_ok=True)
    cache = load_cache(slug)
    todo = [p for p in prompts if p["id"] not in cache or cache[p["id"]].get("error")]
    print(f"{slug}: {len(todo)} to call ({len(prompts) - len(todo)} cached)")
    if not todo:
        return
    lock = threading.Lock()
    f = open(cache_path(slug), "a")
    client = httpx.Client(timeout=180.0, headers={"Authorization": f"Bearer {_key()}"})

    def call(p):
        body = {
            "model": slug, "max_tokens": MAX_TOKENS, "temperature": 0, "stream": True,
            "usage": {"include": True},
            "messages": [{"role": "user", "content": p["prompt"]}],
        }
        row = {"id": p["id"], "error": "exhausted"}
        for attempt in range(4):
            t0, ttft, text, usage = time.perf_counter(), None, [], {}
            try:
                with client.stream("POST", API_URL, json=body) as r:
                    if r.status_code in (429, 500, 502, 503, 529):
                        time.sleep(1.5 * 2**attempt)
                        continue
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        chunk = json.loads(line[6:])
                        delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                        if delta.get("content"):
                            if ttft is None:
                                ttft = (time.perf_counter() - t0) * 1000
                            text.append(delta["content"])
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                row = {
                    "id": p["id"], "text": "".join(text),
                    "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                    "cost_usd": usage.get("cost"), "ttft_ms": round(ttft or -1, 1),
                    "total_ms": round((time.perf_counter() - t0) * 1000, 1),
                }
                break
            except (httpx.HTTPError, json.JSONDecodeError) as e:
                row = {"id": p["id"], "error": f"{type(e).__name__}: {e}"}
                time.sleep(1.5 * 2**attempt)
        with lock:
            f.write(json.dumps(row) + "\n")
            f.flush()
        return row

    done = 0
    with ThreadPoolExecutor(concurrency) as ex:
        for _ in ex.map(call, todo):
            done += 1
            if done % 100 == 0:
                print(f"{slug}: {done}/{len(todo)}")
    f.close()


# ---------- grading + report ----------

def grade_all(prompts: list[dict], slugs: list[str]) -> pd.DataFrame:
    by_id = {p["id"]: p for p in prompts}
    rows = []
    for slug in slugs:
        cache = load_cache(slug)
        for pid, p in by_id.items():
            r = cache.get(pid)
            if r is None or r.get("error"):
                continue
            q = GRADERS[p["source"]](r.get("text", ""), p["gold"])
            rows.append({
                "id": pid, "source": p["source"], "model": slug, "quality": q,
                "cost": r.get("cost_usd") or 0.0, "prompt_tokens": r.get("prompt_tokens"),
                "completion_tokens": r.get("completion_tokens"), "ttft_ms": r.get("ttft_ms"),
                "total_ms": r.get("total_ms"),
            })
    return pd.DataFrame(rows)


def report(slugs: list[str]):
    prompts = load_corpus()
    df = grade_all(prompts, slugs)
    df.to_parquet(os.path.join(DATA, "label_matrix.parquet"))
    print(f"matrix rows: {len(df)} -> data/label_matrix.parquet\n")
    piv_q = df.pivot_table(index="model", columns="source", values="quality", aggfunc="mean")
    piv_q["ALL"] = df.groupby("model")["quality"].mean()
    piv = piv_q.join(df.groupby("model").agg(cost_per_1k=("cost", lambda s: s.mean() * 1000),
                                             ttft_p50=("ttft_ms", "median"),
                                             out_tok=("completion_tokens", "mean")))
    print(piv.round(3).to_string())
    print(f"\ntotal spend so far: ${df.drop_duplicates(['id','model'])['cost'].sum():.2f}")


def pilot(slug: str = "anthropic/claude-haiku-4.5", n: int = 40):
    prompts = load_corpus()
    rng = np.random.default_rng(0)
    sample = [prompts[i] for i in sorted(rng.permutation(len(prompts))[:n].tolist())]
    run_model(slug, sample, concurrency=8)
    df = grade_all(sample, [slug])
    print(df.groupby("source")["quality"].agg(["mean", "count"]).round(3).to_string())
    mean_cost = df["cost"].mean()
    print(f"\npilot mean cost/call ${mean_cost:.5f}; mean out_tokens {df['completion_tokens'].mean():.0f}; "
          f"ttft p50 {df['ttft_ms'].median():.0f}ms")
    # crude full-run projection: scale by output price ratio per model
    prices_out = {"deepseek/deepseek-v4-flash": 0.28, "deepseek/deepseek-v4-pro": 0.87,
                  "moonshotai/kimi-k2.7-code": 3.5, "anthropic/claude-fable-5": 50.0,
                  "anthropic/claude-haiku-4.5": 5.0, "anthropic/claude-opus-5": 25.0,
                  "anthropic/claude-sonnet-5": 10.0}
    est = sum(mean_cost * (p / prices_out[slug]) for p in prices_out.values()) * len(prompts)
    print(f"projected full-matrix spend (rough, output-price scaled): ${est:.0f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "corpus":
        build_corpus()
    elif cmd == "pilot":
        pilot()
    elif cmd == "run":
        slugs = sys.argv[2:] or ROSTER
        prompts = load_corpus()
        for slug in slugs:
            run_model(slug, prompts)
    else:
        report(ROSTER)
