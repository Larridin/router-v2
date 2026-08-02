"""Cross-benchmark: OUR router (calibrated ensemble x dollar-EV) on CODEX's rails.

Runs entirely on the codex project's data, split, embeddings, and reference
policies so the comparison is on their terms:

    uv run --directory /Users/ameya/dev/exp/router-v2/router/codex \
        python /Users/ameya/dev/exp/router-v2/claude/crossbench.py

Protocol: our members fit on TRAIN only; isotonic calibration on VALIDATION
(the split's stated purpose); TEST scored once at the end. Their weave_v075
and codex_centroid are refit with their own code on the same train rows.
Output: claude/results/crossbench.{md,json}.
"""

import json
import re
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from routerlab.centroid import CentroidConfig, fit_centroid_policy
from routerlab.data import load_outcome_matrix
from routerlab.embeddings import EmbeddingCache, FastEmbedEncoder
from routerlab.split import Split, assign_prompt_splits
from routerlab.weave import WeaveConfig, fit_weave_policy

CDX = Path("/Users/ameya/dev/exp/router-v2/router/codex")
OUT = Path("/Users/ameya/dev/exp/router-v2/claude/results")
SEED = 42
BUDGETS = [0.002, 0.005, 0.01, 0.02, 0.04]  # $/request on this matrix's scale
ALPHAS = np.round(np.linspace(0, 1, 21), 3)
LAMBDAS = np.concatenate([[0.0], np.geomspace(0.05, 5000.0, 30)])
BOOT = 2000

# ---------- our router, inlined (mirrors claude/routerlab) ----------

_CODE = re.compile(r"```|\bdef\b|\bclass\b|\breturn\b|\bimport\b|[{};]")
_MATH = re.compile(r"[=+\-*/^<>]|\\frac|\\sum|\d+\.\d+")


def text_features(texts):
    out = np.zeros((len(texts), 12), dtype=np.float32)
    for i, t in enumerate(texts):
        n = max(len(t), 1)
        words = t.split()
        out[i] = [len(t) / 1000.0, np.log1p(len(t)), t.count("\n") + 1, len(words),
                  sum(len(w) for w in words) / max(len(words), 1),
                  sum(c.isdigit() for c in t) / n, sum(ord(c) > 127 for c in t) / n,
                  sum(not c.isalnum() and not c.isspace() for c in t) / n,
                  sum(c.isupper() for c in t) / n,
                  len(_CODE.findall(t)) / n * 1000, len(_MATH.findall(t)) / n * 1000,
                  1.0 if t.rstrip().endswith("?") else 0.0]
    return out


def zscore_rows(q):
    mu, sd = q.mean(axis=1, keepdims=True), q.std(axis=1, keepdims=True)
    out = np.zeros_like(q)
    np.divide(q - mu, sd, out=out, where=sd > 0)
    return out


def shrink(mean, n, global_mean, k0):
    n = np.asarray(n, dtype=np.float64)[:, None]
    return (n * mean + k0 * global_mean[None, :]) / (n + k0)


class ClusterEst:
    def __init__(self, k=64, top_p=4, k0=10.0):
        self.k, self.top_p, self.k0 = k, top_p, k0

    def fit(self, emb, quality, texts=None):
        z = zscore_rows(quality)
        km = KMeans(n_clusters=self.k, n_init=3, random_state=SEED).fit(emb)
        c = km.cluster_centers_
        self.centroids = c / np.clip(np.linalg.norm(c, axis=1, keepdims=True), 1e-12, None)
        means, counts = np.zeros((self.k, quality.shape[1])), np.zeros(self.k)
        for ci in range(self.k):
            m = km.labels_ == ci
            counts[ci] = m.sum()
            if counts[ci]:
                means[ci] = z[m].mean(axis=0)
        self.table = shrink(means, counts, z.mean(axis=0), self.k0)
        return self

    def estimate(self, emb, texts=None):
        top = np.argpartition(-(emb @ self.centroids.T), self.top_p - 1, axis=1)[:, : self.top_p]
        return self.table[top].mean(axis=1)


class KNNEst:
    def __init__(self, k=64, sharp=2.0, k0=10.0):
        self.k, self.sharp, self.k0 = k, sharp, k0

    def fit(self, emb, quality, texts=None):
        self.emb = emb.astype(np.float32)
        self.z = zscore_rows(quality)
        self.gm = self.z.mean(axis=0)
        return self

    def estimate(self, emb, texts=None, batch=1024):
        out = np.empty((len(emb), self.z.shape[1]))
        for i in range(0, len(emb), batch):
            b = emb[i : i + batch].astype(np.float32)
            sims = b @ self.emb.T
            top = np.argpartition(-sims, self.k - 1, axis=1)[:, : self.k]
            w = np.clip(sims[np.arange(len(b))[:, None], top], 0, None) ** self.sharp
            ws = w.sum(axis=1, keepdims=True)
            local = (w[:, :, None] * self.z[top]).sum(axis=1) / np.clip(ws, 1e-12, None)
            neff = np.zeros(len(b))
            np.divide(ws[:, 0] ** 2, (w**2).sum(axis=1), out=neff, where=ws[:, 0] > 0)
            out[i : i + batch] = shrink(local, neff, self.gm, self.k0)
        return out


class GBTEst:
    def fit(self, emb, quality, texts):
        X = np.hstack([emb, text_features(texts)])
        self.models = []
        for m in range(quality.shape[1]):
            g = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                              max_leaf_nodes=63, early_stopping=True,
                                              validation_fraction=0.1, random_state=SEED)
            g.fit(X, quality[:, m])
            self.models.append(g)
        return self

    def estimate(self, emb, texts):
        X = np.hstack([emb, text_features(texts)])
        return np.clip(np.column_stack([g.predict(X) for g in self.models]), 0, 1)


def calibrate(est, cal_emb, cal_q, cal_texts):
    raw = est.estimate(cal_emb, cal_texts)
    isos = []
    for m in range(cal_q.shape[1]):
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        iso.fit(raw[:, m], cal_q[:, m])
        isos.append(iso)
    return isos


def apply_cal(est, isos, emb, texts):
    raw = est.estimate(emb, texts)
    return np.column_stack([iso.predict(raw[:, m]) for m, iso in enumerate(isos)])


class CostModel:
    def fit(self, texts, cost):
        L = np.array([len(t) for t in texts], dtype=np.float64)
        self.coef = [np.polyfit(L, cost[:, m], 1) for m in range(cost.shape[1])]
        self.floor = np.clip(np.quantile(cost, 0.10, axis=0), 1e-9, None)
        return self

    def predict(self, texts):
        L = np.array([len(t) for t in texts], dtype=np.float64)
        out = np.column_stack([b * L + a for b, a in self.coef])
        return np.clip(out, self.floor[None, :], None)


# ---------- evaluation ----------

def realized(choices, quality, cost):
    rows = np.arange(len(choices))
    return quality[rows, choices], cost[rows, choices]


def frontier_points(choices_by_param, quality, cost):
    pts = []
    for param, ch in choices_by_param:
        rq, rc = realized(ch, quality, cost)
        pts.append({"param": float(param), "quality": float(rq.mean()),
                    "cost": float(rc.mean()), "choices": ch})
    return pts


def q_at(pts, budget):
    ok = [p for p in pts if p["cost"] <= budget]
    return max(ok, key=lambda p: p["quality"]) if ok else None


def paired_boot(a, b, n=BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    diffs = a - b
    idx = rng.integers(0, len(diffs), size=(n, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(diffs.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    t0 = time.time()
    matrix = load_outcome_matrix(CDX / ".cache" / "llmrouterbench" / "prepared" / "outcomes.npz")
    enc = FastEmbedEncoder(cache_dir=CDX / ".cache" / "fastembed-models")
    emb = EmbeddingCache(CDX / ".cache" / "embeddings").get_or_encode(matrix.prompts, enc)
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    assign = assign_prompt_splits(matrix.prompt_keys, SEED)
    tr, va, te = (assign == s for s in (Split.TRAIN, Split.VALIDATION, Split.TEST))
    Q, C = matrix.quality.astype(np.float64), matrix.realized_cost.astype(np.float64)
    texts = list(matrix.prompts)
    t_tr = [texts[i] for i in np.flatnonzero(tr)]
    t_va = [texts[i] for i in np.flatnonzero(va)]
    t_te = [texts[i] for i in np.flatnonzero(te)]
    print(f"matrix {Q.shape}; train/val/test = {tr.sum()}/{va.sum()}/{te.sum()} @ {time.time()-t0:.0f}s")

    best_single = int(np.argmax(Q[tr].mean(axis=0)))  # train-selected, their convention
    print(f"train-selected best single: {matrix.models[best_single]}")

    # Their two policies, their code, same train rows.
    weave = fit_weave_policy(emb[tr], Q[tr], C[tr], matrix.models, config=WeaveConfig.v075())
    codex = fit_centroid_policy(emb[tr], Q[tr], C[tr], matrix.models,
                                CentroidConfig(n_clusters=8, top_p=1, shrinkage=1.0,
                                               temperature=0.02, seed=SEED))
    fronts = {}
    fronts["weave_v075"] = frontier_points(
        [(b, weave.select(emb[te], b)) for b in ALPHAS], Q[te], C[te])
    fronts["codex_centroid"] = frontier_points(
        [(b, codex.select_many(emb[te], b)) for b in ALPHAS], Q[te], C[te])
    print(f"their policies swept @ {time.time()-t0:.0f}s")

    # Ours: fit on train, calibrate on validation, score test once.
    members = [ClusterEst(k=64), KNNEst(), GBTEst()]
    cals = []
    for m in members:
        m.fit(emb[tr], Q[tr], t_tr)
        cals.append(calibrate(m, emb[va], Q[va], t_va))
        print(f"fitted {type(m).__name__} @ {time.time()-t0:.0f}s")
    P = np.mean([apply_cal(m, c, emb[te], t_te) for m, c in zip(members, cals)], axis=0)
    cm = CostModel().fit(t_tr, C[tr])
    chat = cm.predict(t_te)
    ours = []
    for lam in LAMBDAS:
        ours.append((lam, np.argmax(P - lam * chat, axis=1)))
    cost_norm = (C[tr].mean(axis=0) - C[tr].mean(axis=0).min())
    cost_norm = cost_norm / max(cost_norm.max(), 1e-12)
    lo = P.min(axis=1, keepdims=True)
    span = np.clip(P.max(axis=1, keepdims=True) - lo, 1e-12, None)
    Pn = (P - lo) / span
    for a in ALPHAS:
        ours.append((a, np.argmax(a * Pn + (1 - a) * (1 - cost_norm[None, :]), axis=1)))
    fronts["ours_ensemble_ev"] = frontier_points(ours, Q[te], C[te])
    print(f"ours swept @ {time.time()-t0:.0f}s")

    # References on test.
    bs_q, bs_c = Q[te, best_single].mean(), C[te, best_single].mean()
    oracle_idx = np.argmax(Q[te] - 1e-12 * C[te], axis=1)
    orc_q, orc_c = realized(oracle_idx, Q[te], C[te])

    # Report.
    lines = ["# Cross-benchmark: our router on Codex's rails (LLMRouterBench)", "",
             f"- test n={int(te.sum())}; train-selected best single `{matrix.models[best_single]}` "
             f"q={bs_q:.3f} @ ${bs_c:.4f}/req; oracle {orc_q.mean():.3f} @ ${orc_c.mean():.4f}/req", "",
             "| router | " + " | ".join(f"q@${b}/req" for b in BUDGETS) + " | parity cost | savings |",
             "|" + "---|" * (len(BUDGETS) + 3)]
    for name, pts in fronts.items():
        cells = []
        for b in BUDGETS:
            p = q_at(pts, b)
            cells.append(f"{p['quality']:.3f}" if p else "—")
        parity = [p["cost"] for p in pts if p["quality"] >= bs_q]
        pc = f"${min(parity):.4f}" if parity else "never"
        sv = f"{1 - min(parity)/bs_c:.0%}" if parity else "—"
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {pc} | {sv} |")

    lines += ["", "## Paired quality diff, ours − weave_v075, at matched budgets", ""]
    for b in BUDGETS:
        po, pw = q_at(fronts["ours_ensemble_ev"], b), q_at(fronts["weave_v075"], b)
        if not (po and pw):
            continue
        aq, _ = realized(po["choices"], Q[te], C[te])
        bq, _ = realized(pw["choices"], Q[te], C[te])
        d, lo_ci, hi_ci = paired_boot(aq, bq)
        verdict = "OURS higher" if lo_ci > 0 else ("weave higher" if hi_ci < 0 else "inconclusive")
        lines.append(f"- ${b}/req: {d:+.4f} [{lo_ci:+.4f}, {hi_ci:+.4f}] -> {verdict}")

    # swe-bench check at weave's bias-0.75-comparable spend.
    ds = np.array(matrix.datasets)[te]
    lines += ["", "## Per-dataset at ~$0.023/req operating point (their bias-0.75 spend)", "",
              "| dataset | ours | weave | codex |", "|---|---:|---:|---:|"]
    pts_at = {k: q_at(v, 0.024) for k, v in fronts.items()}
    for d in sorted(set(ds)):
        m = ds == d
        row = []
        for k in ("ours_ensemble_ev", "weave_v075", "codex_centroid"):
            ch = pts_at[k]["choices"]
            row.append(f"{Q[te][m, ch[m]].mean():.3f}")
        lines.append(f"| {d} | " + " | ".join(row) + " |")

    OUT.mkdir(exist_ok=True)
    for pts in fronts.values():
        for p in pts:
            p["choices"] = None  # drop arrays before JSON
    with open(OUT / "crossbench.json", "w") as f:
        json.dump({"fronts": {k: v for k, v in fronts.items()},
                   "best_single": matrix.models[best_single],
                   "refs": {"best_single_q": float(bs_q), "best_single_c": float(bs_c),
                            "oracle_q": float(orc_q.mean()), "oracle_c": float(orc_c.mean())}}, f, indent=2)
    report = "\n".join(lines)
    (OUT / "crossbench.md").write_text(report + "\n")
    print(report)
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
