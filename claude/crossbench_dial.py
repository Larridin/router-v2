"""Full example prompts + a 0-100 quality dial over our lambda frontier.

Reuses results/crossbench_choices.npz (P-hat, cost predictions, both routers'
choices) — no model fitting. Dial calibration mirrors the Weave scorer's
computeDialCalibration: sweep lambda densely, record every value where the
routed model MIX changes, then space the dial across those breakpoints so
equal dial travel = equal mix change (no dead zones).

    uv run --directory .../router/codex python -P .../claude/crossbench_dial.py
"""

import sys
from pathlib import Path

import numpy as np

CLAUDE = Path("/Users/ameya/dev/exp/router-v2/claude")
from routerlab.data import load_outcome_matrix  # noqa: E402
from routerlab.split import Split, assign_prompt_splits  # noqa: E402

CDX = Path("/Users/ameya/dev/exp/router-v2/router/codex")


def main():
    blob = np.load(CLAUDE / "results" / "crossbench_choices.npz")
    P, chat = blob["P"], blob["chat"]
    w_choice, o_choice = blob["w_choice"], blob["o_choice"]

    matrix = load_outcome_matrix(CDX / ".cache" / "llmrouterbench" / "prepared" / "outcomes.npz")
    assign = assign_prompt_splits(matrix.prompt_keys, 42)
    te = assign == Split.TEST
    te_i = np.flatnonzero(te)
    Q, C = matrix.quality.astype(np.float64), matrix.realized_cost.astype(np.float64)
    texts = list(matrix.prompts)
    ds = np.array(matrix.datasets)[te]
    short = {m: m.split("/")[-1] for m in matrix.models}

    # --- Part 1: full prompts for the same 5 examples (same seed => same picks)
    cats = {}
    rng = np.random.default_rng(3)
    for i in rng.permutation(len(o_choice)):
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

    print("=" * 78)
    for name, i in cats.items():
        gi = te_i[i]
        full = texts[gi]
        body = full if len(full) <= 2500 else full[:2500] + f"\n[... truncated, {len(full)} chars total]"
        print(f"\n### {name}  [{ds[i]}]  weave={short[matrix.models[w_choice[i]]]} "
              f"ours={short[matrix.models[o_choice[i]]]}\n")
        print(body)
        print("-" * 78)

    # --- Part 2: the 0-100 quality dial (mix-change breakpoint calibration)
    lambdas = np.concatenate([[0.0], np.geomspace(0.01, 20000.0, 400)])
    sigs, frontier = [], []
    for lam in lambdas:
        ch = np.argmax(P - lam * chat, axis=1)
        counts = np.bincount(ch, minlength=len(matrix.models))
        sigs.append(counts.tobytes())
        rows = np.arange(len(ch))
        frontier.append((lam, Q[te_i, ch].mean(), C[te_i, ch].mean(), ch))
    # Breakpoints: lambdas where the mix first changes. Dial 100 = lambda 0.
    bp = [0]
    for j in range(1, len(lambdas)):
        if sigs[j] != sigs[j - 1]:
            bp.append(j)
    print(f"\n{len(bp)} mix-change breakpoints across the lambda sweep")

    print("\ndial | lambda | quality | $/req | top models served")
    print("-----|--------|---------|-------|------------------")
    for dial in range(100, -1, -10):
        # dial 100 -> first breakpoint (lambda 0); dial 0 -> last (cheapest mix)
        j = bp[min(int(round((100 - dial) / 100 * (len(bp) - 1))), len(bp) - 1)]
        lam, q, c, ch = frontier[j]
        counts = np.bincount(ch, minlength=len(matrix.models)) / len(ch)
        mix = ", ".join(f"{short[matrix.models[k]]}:{counts[k]:.0%}"
                        for k in np.argsort(-counts)[:3] if counts[k] > 0.01)
        print(f"{dial:4d} | {lam:8.3g} | {q:.3f} | ${c:.4f} | {mix}")


if __name__ == "__main__":
    main()
