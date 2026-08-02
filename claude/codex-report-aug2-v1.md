# Routerlab: What We Have Built So Far

**Date:** August 2, 2026  
**Status:** Research prototype, not yet a production router

## What it does

Routerlab chooses which language model should handle a prompt. Its core flow is:

```text
Prompt
  -> convert the prompt into an embedding
  -> estimate how well each model will answer
  -> estimate each model's cost
  -> balance quality against cost
  -> return the selected model
```

It selects a model, but it does not call that model or proxy the request. Provider calls, retries, authentication, streaming, and other proxy behavior remain separate work.

## What we have done

1. Loaded RouterBench, which contains prompts and measured quality and cost for 11 models.
2. Recreated a Weave-style centroid router as the baseline.
3. Built several alternative quality estimators:
   - a k-nearest-neighbor router;
   - a gradient-boosted predictor;
   - an ensemble combining centroid, kNN, and gradient-boosted estimates.
4. Implemented two model-selection rules:
   - `alpha`: balances normalized quality and cheapness;
   - `dollar-EV`: chooses the model maximizing `predicted quality - lambda * predicted cost`.
5. Built an evaluation harness that compares router quality and realized cost on held-out prompts.
6. Tested many router configurations and generated reports and frontier plots.
7. Collected a smaller label matrix for seven newer models through OpenRouter.
8. Packaged the best experimental RouterBench router as a loadable model bundle.

The main result so far is that the dollar-EV decision rule appears useful at middle budgets. This is promising evidence, but it is not yet a definitive claim that the new router broadly beats Weave.

## How to use it

From the project directory, install the locked environment and run the tests:

```bash
cd ~/dev/exp/router-v2/claude
uv sync --frozen
uv run pytest -q
```

Run the basic Weave-style-versus-kNN benchmark:

```bash
uv run python -m routerlab.run
```

This writes metrics, a report, and a frontier plot under `results/`.

Route one prompt using the saved `v0.2` bundle:

```bash
uv run python -m routerlab.bundle route \
  "Explain why the sky is blue" \
  --lam 30
```

Example output:

```json
{
  "model": "gpt-4-1106-preview",
  "p_success": 0.82,
  "est_cost_usd": 0.00079,
  "lambda": 30
}
```

The `lambda` value controls the trade-off:

- `0` maximizes predicted quality.
- Higher values increasingly prefer cheaper models.

## Current limitations

- The saved bundle uses RouterBench's older 11-model roster.
- The seven-current-model work evaluates routing ideas but has not yet produced a production-ready bundle.
- The result is an exploratory benchmark, not yet a clean final comparison against Weave.
- The router only returns a model choice; it is not integrated into the production proxy.
- The MBPP/current-roster grading workflow should not be rerun until its code-execution grader is sandboxed.
