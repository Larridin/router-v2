"""Deterministic Markdown rendering for routing experiment evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_report(
    search: dict[str, Any],
    test: dict[str, Any],
    provenance: dict[str, Any],
) -> str:
    """Render search and held-out results without adding ambient state."""

    selected = search["selected"]
    config = selected["config"]
    lines = [
        "# Codex Centroid Router Evaluation",
        "",
        "This report evaluates a frozen semantic-centroid routing policy against "
        "train-only baselines on a prompt-hash held-out test split.",
        "",
        "## Provenance",
        "",
        f"- Artifact: `{test['artifact_id']}`",
        f"- Search seed: `{search['seed']}`",
    ]
    for key in sorted(provenance):
        lines.append(f"- {key.replace('_', ' ').title()}: `{_scalar(provenance[key])}`")

    counts = search["split_counts"]
    lines.extend(
        [
            "",
            "## Data split",
            "",
            "| Train | Validation | Test |",
            "| ---: | ---: | ---: |",
            f"| {counts['train']} | {counts['validation']} | {counts['test']} |",
            "",
            "Prompt text is normalized, SHA-256 hashed, and assigned 70/15/15. "
            "Policy fitting and hyperparameter selection never use test outcomes; "
            "changing every test outcome leaves the selected policy tensors and artifact "
            "ID identical in a regression test.",
            "",
            "## Selected candidate",
            "",
            f"Selection metric: {search['selection_metric']}.",
            "",
            "| Clusters | Top-p | Shrinkage | Temperature | Seed | Validation objective |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {config['n_clusters']} | {config['top_p']} | "
            f"{_number(config['shrinkage'])} | {_number(config['temperature'])} | "
            f"{config['seed']} | {_number(selected['objective'])} |",
        ]
    )
    lines.extend(_weave_comparison_lines(test))
    lines.extend(
        [
            "",
            "## Held-out comparison",
            "",
            f"Test prompts: {test['test_rows']}. The table uses quality bias 0.75. "
            "Micro weights prompts equally; macro weights datasets equally.",
            "",
            "| Policy | Bias | Micro quality | Macro quality | Micro cost (USD/request) | "
            "Macro cost | Gain vs best single | Oracle gap | "
            "Normalized regret | Model entropy |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    comparison_points = sorted(
        (point for point in test["points"] if point["quality_bias"] == 0.75),
        key=lambda point: point["name"],
    )
    for point in comparison_points:
        lines.append(_point_row(point))

    saving = test.get("candidate_cost_saving_at_best_single_quality")
    saving_text = "not achieved" if saving is None else f"{100 * saving:.2f}%"
    macro_saving = test.get("macro_candidate_cost_saving_at_best_single_quality")
    macro_saving_text = "not achieved" if macro_saving is None else f"{100 * macro_saving:.2f}%"
    weave_saving = test.get("weave_cost_saving_at_best_single_quality")
    weave_saving_text = "not achieved" if weave_saving is None else f"{100 * weave_saving:.2f}%"
    macro_weave_saving = test.get("macro_weave_cost_saving_at_best_single_quality")
    macro_weave_saving_text = (
        "not achieved" if macro_weave_saving is None else f"{100 * macro_weave_saving:.2f}%"
    )
    lines.extend(
        [
            "",
            f"Candidate micro cost saving at best-single micro quality: **{saving_text}**.",
            f"Candidate macro cost saving at best-single macro quality: **{macro_saving_text}**.",
            f"Weave micro cost saving at best-single micro quality: **{weave_saving_text}**.",
            f"Weave macro cost saving at best-single macro quality: **{macro_weave_saving_text}**.",
            f"Normalized frontier area: **{_number(test['normalized_frontier_area'])}**.",
            "",
            "## Pareto frontier",
            "",
            "| Policy | Bias | Micro quality | Macro quality | Micro cost (USD/request) | "
            "Macro cost | Gain vs best single | Oracle gap | "
            "Normalized regret | Model entropy |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for point in test["frontier"]:
        lines.append(_point_row(point))

    lines.extend(["", "## Paired bootstrap", ""])
    for name, interval in sorted(test.get("bootstrap", {}).items()):
        confidence = 100 * interval["confidence"]
        lines.append(
            f"- {name.replace('_', ' ')}: {_number(interval['estimate'])}; "
            f"{confidence:.0f}% CI [{_number(interval['low'])}, "
            f"{_number(interval['high'])}] over {interval['replicates']} paired resamples."
        )
    if test.get("bootstrap_by_bias"):
        lines.extend(["", "Quality difference versus best single by bias:", ""])
        for bias, comparisons in sorted(
            test["bootstrap_by_bias"].items(), key=lambda item: float(item[0])
        ):
            interval = comparisons["quality_vs_best_single"]
            lines.append(
                f"- Bias {bias}: {_number(interval['estimate'])}; "
                f"{100 * interval['confidence']:.0f}% CI "
                f"[{_number(interval['low'])}, {_number(interval['high'])}]."
            )

    lines.extend(
        [
            "",
            "## Per-dataset candidate results",
            "",
            "| Dataset | Bias | Quality | Cost | Oracle gap |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for point in sorted(
        test.get("per_dataset", []),
        key=lambda value: (value["dataset"], value["quality_bias"]),
    ):
        lines.append(
            f"| {point['dataset']} | {_number(point['quality_bias'])} | "
            f"{_number(point['mean_quality'])} | {_number(point['mean_cost'])} | "
            f"{_number(point['oracle_quality_gap'])} |"
        )

    lines.extend(
        [
            "",
            "## Robustness and latency",
            "",
            _robustness_summary(test.get("robustness")),
            _latency_summary(test.get("latency")),
            "",
            "## Caveats",
            "",
            "- These are offline counterfactual choices over previously collected model "
            "outputs, not live provider calls.",
            "- Historical prices, model versions, benchmark prompts, and benchmark scoring "
            "may not represent production traffic.",
            "- Hyperparameters were selected on validation results; only the frozen winner "
            "was summarized on test.",
            "- Weave uses the fixed v0.75 hyperparameters without public-data tuning; "
            "Codex searched 144 configurations, so the tuning budget favors Codex.",
            "- Oracle uses held-out outcomes and is an evaluator-only upper bound, not a "
            "servable router.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "cd codex",
            "uv sync --frozen",
            "uv run routerlab run-all --cache-root .cache --artifact-root artifacts "
            "--results-root results --seed 42",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(results_root: Path) -> Path:
    """Load deterministic result JSON and write REPORT.md."""

    root = Path(results_root)
    search = _read_object(root / "search.json")
    test = _read_object(root / "test_metrics.json")
    provenance = _read_object(root / "provenance.json")
    _validate_result_context(search, test, provenance)
    for key in ("robustness", "latency"):
        path = root / f"{key}.json"
        if path.exists():
            test[key] = _read_object(path)
            _validate_diagnostic_context(key, test[key], test)
    output = root / "REPORT.md"
    output.write_text(render_report(search, test, provenance), encoding="utf-8")
    return output


def _validate_result_context(
    search: dict[str, Any],
    test: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    seeds = (search.get("seed"), test.get("seed"), provenance.get("split_seed"))
    if not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds):
        raise ValueError("result files must contain valid split seeds")
    if len(set(seeds)) != 1:
        raise ValueError("result split seeds do not match")
    if search.get("artifact_id") != test.get("artifact_id"):
        raise ValueError("search and test artifact IDs do not match")
    if search.get("split_counts") != provenance.get("split_counts"):
        raise ValueError("search and provenance split counts do not match")
    if test.get("matrix_sha256") != provenance.get("matrix_sha256"):
        raise ValueError("test and provenance matrix SHA-256 values do not match")


def _validate_diagnostic_context(
    name: str,
    diagnostic: object,
    test: dict[str, Any],
) -> None:
    if not isinstance(diagnostic, dict):
        raise ValueError(f"{name} result must be an object")
    for field, expected in (
        ("artifact_id", test.get("artifact_id")),
        ("matrix_sha256", test.get("matrix_sha256")),
        ("seed", test.get("seed")),
    ):
        if diagnostic.get(field) != expected:
            raise ValueError(f"{name} {field} does not match test evaluation")


def _weave_comparison_lines(test: dict[str, Any]) -> list[str]:
    baseline = test["weave_baseline"]
    config = baseline["config"]
    lines = [
        "",
        "## Codex versus Weave algorithm",
        "",
        "This is an algorithm-level comparison: both routers use the same public rows, "
        "BGE embeddings, model roster, split, train-selected best-single reference, and "
        "held-out outcomes.",
        "",
        "The shared sweep uses quality bias directly as uniform alpha. Production "
        "roster-specific dial calibration and per-cluster alpha floors are outside "
        "this core comparison, as are provider/capability filters and subsidies.",
        "",
        f"Weave recipe: K={config['n_clusters']}, top-P={config['top_p']}, "
        f"shrinkage={_number(config['shrinkage'])}, seed={config['seed']}; "
        f"policy SHA-256 `{baseline['policy_sha256']}`.",
        f"Trainer source revision: `{baseline['trainer_source_revision']}`.",
        "",
        "| Bias | Codex gain vs best single | Weave gain vs best single | "
        "Codex-Weave quality | 95% paired quality CI | Codex cost | Weave cost | "
        "Codex-Weave cost | 95% paired cost CI | Quality verdict |",
        "| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    comparisons = sorted(test["weave_comparison"], key=lambda value: value["quality_bias"])
    for comparison in comparisons:
        interval = comparison["quality_bootstrap"]
        cost_interval = comparison["cost_bootstrap"]
        verdict = _quality_verdict(interval["low"], interval["high"])
        lines.append(
            f"| {_number(comparison['quality_bias'])} | "
            f"{_number(comparison['codex_quality_gain_vs_best_single'])} | "
            f"{_number(comparison['weave_quality_gain_vs_best_single'])} | "
            f"{_number(comparison['codex_minus_weave_quality'])} | "
            f"[{_number(interval['low'])}, {_number(interval['high'])}] | "
            f"{_number(comparison['codex_mean_cost'])} | "
            f"{_number(comparison['weave_mean_cost'])} | "
            f"{_number(comparison['codex_minus_weave_cost'])} | "
            f"[{_number(cost_interval['low'])}, {_number(cost_interval['high'])}] | "
            f"{verdict} |"
        )
    lines.extend(["", "Paired quality gain versus best single:", ""])
    for comparison in comparisons:
        codex = comparison["codex_quality_bootstrap_vs_best_single"]
        weave = comparison["weave_quality_bootstrap_vs_best_single"]
        lines.append(
            f"- Bias {_number(comparison['quality_bias'])}: "
            f"Codex {_number(codex['estimate'])}, 95% CI "
            f"[{_number(codex['low'])}, {_number(codex['high'])}]; "
            f"Weave {_number(weave['estimate'])}, 95% CI "
            f"[{_number(weave['low'])}, {_number(weave['high'])}]."
        )
    return lines


def _quality_verdict(low: object, high: object) -> str:
    if float(low) > 0:
        return "Codex higher"
    if float(high) < 0:
        return "Weave higher"
    return "Inconclusive"


def _point_row(point: dict[str, Any]) -> str:
    return (
        f"| {point['name']} | {_number(point['quality_bias'])} | "
        f"{_number(point['mean_quality'])} | "
        f"{_number(point.get('macro_mean_quality', point['mean_quality']))} | "
        f"{_number(point['mean_cost'])} | "
        f"{_number(point.get('macro_mean_cost', point['mean_cost']))} | "
        f"{_number(point['best_single_quality_gain'])} | "
        f"{_number(point['oracle_quality_gap'])} | "
        f"{_number(point['mean_normalized_oracle_regret'])} | "
        f"{_number(point['normalized_model_entropy'])} |"
    )


def _robustness_summary(value: object) -> str:
    if value is None:
        return "- Robustness: not measured in this run."
    if not isinstance(value, dict):
        raise ValueError("robustness result must be an object")
    return (
        f"- Robustness: {value['sample_size']} prompts x "
        f"{len(value['perturbations'])} perturbations; "
        f"flip rate {_number(value['overall_flip_rate'])}, mean quality delta "
        f"{_number(value['mean_quality_delta'])}, mean cost delta "
        f"{_number(value['mean_cost_delta'])}."
    )


def _latency_summary(value: object) -> str:
    if value is None:
        return "- Latency: not measured in this run."
    if not isinstance(value, dict):
        raise ValueError("latency result must be an object")
    return (
        f"- Latency: pure Python vector scorer over {value['routes']} routes, excluding "
        f"embedding; p50 {_number(value['p50_microseconds'])} µs, "
        f"p95 {_number(value['p95_microseconds'])} µs, "
        f"p99 {_number(value['p99_microseconds'])} µs."
    )


def _number(value: object) -> str:
    return f"{float(value):.6f}"


def _scalar(value: object) -> str:
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value
