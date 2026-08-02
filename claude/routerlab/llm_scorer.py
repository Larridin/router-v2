"""H4: a cheap LLM as routing-signal source, via OpenRouter.

One tiny, latency-bounded call per prompt: the LLM emits a JSON verdict
(difficulty 0-10, domain, needs-reasoning). Per-call wall-clock latency is
recorded because an LLM in the routing path adds to every request's TTFT —
the verdict must be quality-per-millisecond, not quality alone.

Results are cached to a JSONL keyed by prompt hash, so pilots roll into full
runs and reruns are free.
"""

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
KEY_PATH = os.path.join(ROOT, ".openrouter_key")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"

DOMAINS = ["code", "math", "knowledge", "commonsense", "chinese", "other"]
MAX_PROMPT_CHARS = 1500

SYSTEM = (
    "You are a routing classifier. The user message contains a prompt inside "
    "<prompt-to-rate> tags. NEVER answer or solve that prompt — only rate it."
)

# Restated AFTER the prompt (recency wins when the rated prompt itself contains
# instructions like "answer with A/B/C/D"), plus an assistant prefill to force
# JSON continuation.
INSTRUCTION = (
    "Do NOT answer the prompt above. Rate it for model routing. Reply with ONLY this JSON: "
    '{"difficulty": <0-10 int>, "domain": "code|math|knowledge|commonsense|chinese|other", '
    '"reasoning": <true|false>}'
)
PREFILL = '{"difficulty":'
PROMPT_VERSION = "v2"

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)
DEFAULT_SCORE = {"difficulty": 5, "domain": "other", "reasoning": False}


def parse_score(text: str) -> tuple[dict, bool]:
    """Extract the JSON verdict; fall back to neutral defaults on garbage."""
    m = _JSON_RE.search(text or "")
    if not m:
        return dict(DEFAULT_SCORE), False
    try:
        raw = json.loads(m.group(0))
        d = int(raw.get("difficulty", 5))
        dom = str(raw.get("domain", "other")).lower().strip()
        return {
            "difficulty": min(max(d, 0), 10),
            "domain": dom if dom in DOMAINS else "other",
            "reasoning": bool(raw.get("reasoning", False)),
        }, True
    except (ValueError, TypeError, json.JSONDecodeError):
        return dict(DEFAULT_SCORE), False


def _key() -> str:
    return open(KEY_PATH).read().strip()


def _cache_path(model: str) -> str:
    slug = model.replace("/", "_").replace(".", "_")
    return os.path.join(ROOT, "data", f"llm_scores_{slug}_{PROMPT_VERSION}.jsonl")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_cache(model: str) -> dict:
    path = _cache_path(model)
    cache = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                cache[row["h"]] = row
    return cache


class Scorer:
    def __init__(self, model: str = DEFAULT_MODEL, concurrency: int = 32):
        self.model = model
        self.concurrency = concurrency
        self.cache = load_cache(model)
        self._lock = threading.Lock()
        self._cache_f = open(_cache_path(model), "a")
        self.client = httpx.Client(
            timeout=30.0,
            headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"},
        )

    def _call(self, text: str) -> dict:
        h = _hash(text)
        if h in self.cache:
            return self.cache[h]
        body = {
            "model": self.model,
            "max_tokens": 50,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": f"<prompt-to-rate>\n{text[-MAX_PROMPT_CHARS:]}\n</prompt-to-rate>\n\n{INSTRUCTION}",
                },
                {"role": "assistant", "content": PREFILL},
            ],
        }
        row = None
        for attempt in range(5):
            t0 = time.perf_counter()
            try:
                r = self.client.post(API_URL, json=body)
                latency_ms = (time.perf_counter() - t0) * 1000
                if r.status_code in (429, 500, 502, 503, 529):
                    time.sleep(0.5 * 2**attempt)
                    continue
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"] or ""
                if not content.lstrip().startswith("{"):
                    content = PREFILL + content  # provider consumed the prefill
                score, ok = parse_score(content)
                row = {"h": h, **score, "ok": ok, "latency_ms": round(latency_ms, 1)}
                break
            except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError):
                time.sleep(0.5 * 2**attempt)
        if row is None:
            row = {"h": h, **DEFAULT_SCORE, "ok": False, "latency_ms": -1.0}
        with self._lock:
            self.cache[h] = row
            self._cache_f.write(json.dumps(row) + "\n")
            self._cache_f.flush()
        return row

    def score(self, texts: list[str]) -> list[dict]:
        with ThreadPoolExecutor(self.concurrency) as ex:
            return list(ex.map(self._call, texts))


def features(rows: list[dict]) -> np.ndarray:
    """[difficulty/10, reasoning] + domain one-hot -> (n, 8)."""
    out = np.zeros((len(rows), 2 + len(DOMAINS)), dtype=np.float32)
    for i, r in enumerate(rows):
        out[i, 0] = r["difficulty"] / 10.0
        out[i, 1] = 1.0 if r["reasoning"] else 0.0
        out[i, 2 + DOMAINS.index(r["domain"])] = 1.0
    return out


def latency_stats(rows: list[dict]) -> dict:
    lat = np.array([r["latency_ms"] for r in rows if r.get("latency_ms", -1) > 0])
    fails = sum(1 for r in rows if not r.get("ok"))
    if len(lat) == 0:
        return {"n_timed": 0, "parse_fail_rate": fails / max(len(rows), 1)}
    return {
        "n_timed": int(len(lat)),
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "mean_ms": float(lat.mean()),
        "parse_fail_rate": fails / max(len(rows), 1),
    }
