from __future__ import annotations

import json
from pathlib import Path

import pytest

from routerlab.report import render_report, write_report


def test_report_renders_reproducible_evidence_and_caveats() -> None:
    search = {
        "seed": 42,
        "split_counts": {"train": 70, "validation": 15, "test": 15},
        "selection_metric": "mean validation normalized oracle regret",
        "selected": {
            "objective": 0.1234567,
            "config": {
                "n_clusters": 16,
                "top_p": 2,
                "shrinkage": 5.0,
                "temperature": 0.05,
                "seed": 42,
            },
        },
    }
    candidate = {
        "name": "codex_centroid",
        "quality_bias": 0.75,
        "mean_quality": 0.8,
        "mean_cost": 0.2,
        "best_single_quality_gain": 0.03,
        "oracle_quality_gap": 0.1,
        "mean_normalized_oracle_regret": 0.12,
        "normalized_model_entropy": 0.7,
        "model_counts": [8, 7],
    }
    best = {**candidate, "name": "best_single", "mean_quality": 0.77, "mean_cost": 0.3}
    weave = {
        **candidate,
        "name": "weave_v075",
        "mean_quality": 0.79,
        "mean_cost": 0.24,
        "best_single_quality_gain": 0.02,
    }
    oracle = {**candidate, "name": "oracle", "mean_quality": 0.9, "mean_cost": 0.25}
    test = {
        "artifact_id": "abc123",
        "test_rows": 15,
        "points": [candidate, best, weave, oracle],
        "frontier": [candidate],
        "oracle_points": [oracle],
        "normalized_frontier_area": 0.55,
        "candidate_cost_saving_at_best_single_quality": 1 / 3,
        "macro_candidate_cost_saving_at_best_single_quality": 0.25,
        "weave_cost_saving_at_best_single_quality": 0.2,
        "macro_weave_cost_saving_at_best_single_quality": 0.1,
        "weave_baseline": {
            "name": "weave_v075",
            "algorithm": "Weave v0.75 core cluster recipe retrained on shared public data",
            "source": "internal/router/cluster/artifacts/v0.75/metadata.yaml",
            "trainer_source_revision": "b74af8941cb71e3a0bc43af9d9a68b0c727fb7cf",
            "config": {
                "n_clusters": 16,
                "top_p": 4,
                "shrinkage": 10.0,
                "seed": 42,
                "n_init": 10,
            },
            "policy_sha256": "d" * 64,
        },
        "weave_comparison": [
            {
                "quality_bias": 0.75,
                "codex_mean_quality": 0.8,
                "weave_mean_quality": 0.79,
                "codex_mean_cost": 0.2,
                "weave_mean_cost": 0.24,
                "codex_quality_gain_vs_best_single": 0.03,
                "weave_quality_gain_vs_best_single": 0.02,
                "codex_minus_weave_quality": 0.01,
                "codex_minus_weave_cost": -0.04,
                "codex_quality_bootstrap_vs_best_single": {
                    "estimate": 0.03,
                    "low": 0.01,
                    "high": 0.05,
                    "confidence": 0.95,
                    "replicates": 2000,
                },
                "weave_quality_bootstrap_vs_best_single": {
                    "estimate": 0.02,
                    "low": -0.005,
                    "high": 0.04,
                    "confidence": 0.95,
                    "replicates": 2000,
                },
                "quality_bootstrap": {
                    "estimate": 0.01,
                    "low": -0.01,
                    "high": 0.03,
                    "confidence": 0.95,
                    "replicates": 2000,
                },
                "cost_bootstrap": {
                    "estimate": -0.04,
                    "low": -0.06,
                    "high": -0.02,
                    "confidence": 0.95,
                    "replicates": 2000,
                },
            }
        ],
        "bootstrap_bias": 0.75,
        "bootstrap": {
            "quality_vs_best_single": {
                "estimate": 0.03,
                "low": 0.01,
                "high": 0.05,
                "confidence": 0.95,
                "replicates": 2000,
            }
        },
        "per_dataset": [{"dataset": "math", **candidate}],
    }
    provenance = {
        "dataset": "NPULH/LLMRouterBench",
        "archive_sha256": "b79f8cde",
        "license": "MIT repository; dataset archive has no separate card license",
    }

    report = render_report(search, test, provenance)

    assert report.startswith("# Codex Centroid Router Evaluation\n")
    assert "`abc123`" in report
    assert "16 | 2 | 5.000000 | 0.050000" in report
    assert "codex_centroid | 0.750000 | 0.800000 | 0.800000 | 0.200000" in report
    assert "## Codex versus Weave algorithm" in report
    assert "same public rows, BGE embeddings, model roster, split" in report
    assert "uses quality bias directly as uniform alpha" in report
    assert "dial calibration and per-cluster alpha floors are outside" in report
    assert "b74af8941cb71e3a0bc43af9d9a68b0c727fb7cf" in report
    assert "0.030000 | 0.020000 | 0.010000 | [-0.010000, 0.030000]" in report
    assert "[-0.060000, -0.020000]" in report
    assert "Weave 0.020000, 95% CI [-0.005000, 0.040000]" in report
    assert "Weave macro cost saving at best-single macro quality: **10.00%**." in report
    assert "Weave uses the fixed v0.75 hyperparameters without public-data tuning" in report
    assert "Inconclusive" in report
    assert "95% CI [0.010000, 0.050000]" in report
    assert "NPULH/LLMRouterBench" in report
    assert "Policy fitting and hyperparameter selection never use test outcomes" in report
    assert "uv run routerlab run-all" in report
    assert report == render_report(search, test, provenance)


def test_report_rejects_results_from_a_different_split(tmp_path: Path) -> None:
    values = {
        "search.json": {"seed": 42, "split_counts": {"train": 7, "validation": 1, "test": 2}},
        "test_metrics.json": {"seed": 43, "matrix_sha256": "a" * 64},
        "provenance.json": {
            "split_seed": 42,
            "split_counts": {"train": 7, "validation": 1, "test": 2},
            "matrix_sha256": "a" * 64,
        },
    }
    for name, value in values.items():
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="split seed"):
        write_report(tmp_path)


def test_report_rejects_diagnostics_from_another_artifact(tmp_path: Path) -> None:
    split_counts = {"train": 7, "validation": 1, "test": 2}
    values = {
        "search.json": {
            "artifact_id": "a" * 64,
            "seed": 42,
            "split_counts": split_counts,
        },
        "test_metrics.json": {
            "artifact_id": "a" * 64,
            "seed": 42,
            "matrix_sha256": "b" * 64,
        },
        "provenance.json": {
            "split_seed": 42,
            "split_counts": split_counts,
            "matrix_sha256": "b" * 64,
        },
        "latency.json": {
            "artifact_id": "c" * 64,
            "seed": 42,
            "matrix_sha256": "b" * 64,
        },
    }
    for name, value in values.items():
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="latency artifact_id"):
        write_report(tmp_path)
