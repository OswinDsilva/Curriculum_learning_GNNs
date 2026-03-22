from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from experiments.aggregate_results import (
    collect_result_rows,
    compute_significance,
    summarise_rows,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_collect_and_summarise_results() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_json(
            root / "baseline" / "cora_gcn_seed0.json",
            {
                "config": {
                    "dataset": "cora",
                    "model": "gcn",
                    "seed": 0,
                    "experiment_condition": "baseline",
                },
                "condition": "baseline",
                "standard": {"auc": 0.90, "mrr": 0.20},
                "heart": {"heart_mrr": 0.10},
            },
        )
        _write_json(
            root / "baseline" / "cora_gcn_seed1.json",
            {
                "config": {
                    "dataset": "cora",
                    "model": "gcn",
                    "seed": 1,
                    "experiment_condition": "baseline",
                },
                "condition": "baseline",
                "standard": {"auc": 0.92, "mrr": 0.22},
                "heart": {"heart_mrr": 0.12},
            },
        )

        rows = collect_result_rows(root)
        summary = summarise_rows(rows)

        auc_row = next(row for row in summary if row["metric"] == "auc")
        assert auc_row["mean"] == 0.91
        assert auc_row["n_seeds"] == 2


def test_compute_significance_pairs_baseline_and_curriculum() -> None:
    rows = [
        {
            "category": "baseline",
            "dataset": "cora",
            "model": "gcn",
            "heuristic": None,
            "condition": "baseline",
            "seed": 0,
            "metrics": {"heart_mrr": 0.10, "auc": 0.90},
        },
        {
            "category": "baseline",
            "dataset": "cora",
            "model": "gcn",
            "heuristic": None,
            "condition": "baseline",
            "seed": 1,
            "metrics": {"heart_mrr": 0.11, "auc": 0.91},
        },
        {
            "category": "curriculum",
            "dataset": "cora",
            "model": "gcn",
            "heuristic": "cn",
            "condition": "curriculum",
            "seed": 0,
            "metrics": {"heart_mrr": 0.20, "auc": 0.93},
        },
        {
            "category": "curriculum",
            "dataset": "cora",
            "model": "gcn",
            "heuristic": "cn",
            "condition": "curriculum",
            "seed": 1,
            "metrics": {"heart_mrr": 0.22, "auc": 0.94},
        },
    ]

    significance = compute_significance(rows)
    metric_names = {row["metric"] for row in significance}
    assert "heart_mrr" in metric_names
    assert "auc" in metric_names


def test_aggregate_cli_writes_csv_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_json(
            root / "baseline" / "cora_gcn_seed0.json",
            {
                "config": {
                    "dataset": "cora",
                    "model": "gcn",
                    "seed": 0,
                    "experiment_condition": "baseline",
                },
                "condition": "baseline",
                "standard": {"auc": 0.90},
                "heart": {},
            },
        )
        out_dir = root / "summaries"

        from experiments.aggregate_results import main
        import sys

        old_argv = sys.argv
        sys.argv = [
            "aggregate_results.py",
            "--results_dir",
            str(root),
            "--output_dir",
            str(out_dir),
        ]
        try:
            main()
        finally:
            sys.argv = old_argv

        baseline_csv = out_dir / "baseline_summary.csv"
        assert baseline_csv.exists()
        rows = list(csv.DictReader(baseline_csv.open()))
        assert rows[0]["metric"] == "auc"
