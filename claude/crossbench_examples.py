"""Concrete example routes: weave vs ours on real LLMRouterBench test prompts.

Refits both policies exactly as crossbench.py, picks the matched-cost
operating points, then prints 5 illustrative examples by category:
same-pick, ours-cheaper-both-right, ours-cheaper-and-WRONG (the perceived-
quality failure mode), ours-right-weave-wrong, and an agentic (swe-bench).
Also saves P-hat / choices to npz for later reuse.

    uv run --directory .../router/codex python -P .../claude/crossbench_examples.py
"""

import json
from pathlib import Path

import numpy as np

CLAUDE = Path("/Users/ameya/dev/exp/router-v2/claude")
import sys

sys.path.append(str(CLAUDE))  # append, not insert: their routerlab must win the name
from crossbench import (  # noqa: E402  (reuse the exact same fitting code)
    CDX, ClusterEst, CostModel, GBTEst, KNNEst, LAMBDAS, apply_cal, calibrate, realized,
)
from routerlab.data import load_outcome_matrix  # noqa: E402
from routerlab.embeddings import EmbeddingCache, FastEmbedEncoder  # noqa: E402
from routerlab.split import Split, assign_prompt_splits  # noqa: E402
from routerlab.weave import WeaveConfig, fit_weave_policy  # noqa: E402

SEED = 42


def main():
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

    weave = fit_weave_policy(emb[tr], Q[tr], C[tr], matrix.models, config=WeaveConfig.v075())
    w_choice = weave.select(emb[te], 0.75)
    w_q, w_c = realized(w_choice, Q[te], C[te])
    w_cost = w_c.mean()
    print(f"weave bias=0.75 realized cost ${w_cost:.4f}/req")

    members = [ClusterEst(k=64), KNNEst(), GBTEst()]
    cals = []
    for m in members:
        m.fit(emb[tr], Q[tr], t_tr)
        cals.append(calibrate(m, emb[va], Q[va], t_va))
    P = np.mean([apply_cal(m, c, emb[te], t_te) for m, c in zip(members, cals)], axis=0)
    chat = CostModel().fit(t_tr, C[tr]).predict(t_te)

    # Matched-cost operating point: largest-lambda frontier point with cost <= weave's.
    best = None
    for lam in LAMBDAS:
        ch = np.argmax(P - lam * chat, axis=1)
        cost = realized(ch, Q[te], C[te])[1].mean()
        if cost <= w_cost and (best is None or lam < best[0]):
            best = (lam, ch, cost)
    lam, o_choice, o_cost = best
    o_q, o_c = realized(o_choice, Q[te], C[te])
    print(f"ours lambda={lam:.3g} realized cost ${o_cost:.4f}/req; "
          f"quality ours {o_q.mean():.3f} vs weave {w_q.mean():.3f}")

    np.savez(CLAUDE / "results" / "crossbench_choices.npz",
             P=P, chat=chat, w_choice=w_choice, o_choice=o_choice, lam=lam)

    ds = np.array(matrix.datasets)[te]
    te_i = np.flatnonzero(te)
    short = {m: m.split("/")[-1] for m in matrix.models}

    def row(i):
        gi = te_i[i]
        wm, om = w_choice[i], o_choice[i]
        return {
            "dataset": ds[i],
            "prompt": texts[gi][:220].replace("\n", " "),
            "weave_model": short[matrix.models[wm]],
            "weave_correct": bool(Q[gi, wm] >= 0.5),
            "weave_cost": float(C[gi, wm]),
            "ours_model": short[matrix.models[om]],
            "ours_correct": bool(Q[gi, om] >= 0.5),
            "ours_cost": float(C[gi, om]),
            "ours_phat": float(P[i, om]),
        }

    cats = {}
    rng = np.random.default_rng(3)
    idx = rng.permutation(len(o_choice))
    for i in idx:
        wq, oq = Q[te_i[i], w_choice[i]], Q[te_i[i], o_choice[i]]
        wc, oc = C[te_i[i], w_choice[i]], C[te_i[i], o_choice[i]]
        if "same_pick" not in cats and w_choice[i] == o_choice[i] and oq >= 0.5:
            cats["same_pick"] = i
        if "ours_cheaper_both_right" not in cats and oc < wc * 0.5 and oq >= 0.5 and wq >= 0.5:
            cats["ours_cheaper_both_right"] = i
        if "ours_cheaper_and_wrong" not in cats and oc < wc and oq < 0.5 and wq >= 0.5:
            cats["ours_cheaper_and_wrong"] = i
        if "ours_right_weave_wrong" not in cats and oq >= 0.5 and wq < 0.5:
            cats["ours_right_weave_wrong"] = i
        if "agentic_swebench" not in cats and ds[i] == "swe-bench" and oq >= 0.5 and wq < 0.5:
            cats["agentic_swebench"] = i

    out = {name: row(i) for name, i in cats.items()}
    print(json.dumps(out, indent=2))

    # Aggregate framing for the quality-consciousness discussion.
    downgraded = (chat[np.arange(len(o_choice)), o_choice] <
                  chat[np.arange(len(w_choice)), w_choice] * 0.5)
    n_down = int(downgraded.sum())
    regret = int(((o_q < 0.5) & (w_q >= 0.5) & downgraded).sum())
    upgraded_win = int(((o_q >= 0.5) & (w_q < 0.5)).sum())
    print(f"\nOf {len(o_choice)} test prompts at matched spend: ours routes {n_down} "
          f"to a >=2x-cheaper model than weave; {regret} of those lose a win weave had "
          f"({regret/max(n_down,1):.1%} downgrade regret); ours wins {upgraded_win} prompts weave missed.")


if __name__ == "__main__":
    main()
