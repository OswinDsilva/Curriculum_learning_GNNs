from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ttest_rel


SUMMARY_FILES = {
    "baseline": "baseline_summary.csv",
    "curriculum": "curriculum_summary.csv",
    "ablation": "ablation_summary.csv",
}


def infer_category(path: Path, payload: dict[str, Any]) -> str:
    condition = str(
        payload.get("condition")
        or payload.get("config", {}).get("experiment_condition")
        or ""
    )
    if "ablation" in path.parts or condition.startswith("abl-"):
        return "ablation"
    if "baseline" in path.parts or condition == "baseline":
        return "baseline"
    return "curriculum"


def flatten_metrics(payload: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for section in ("standard", "heart"):
        values = payload.get(section, {}) or {}
        for key, value in values.items():
            if value is None:
                continue
            flat[key] = float(value)
    return flat


def collect_result_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.rglob("*.json")):
        if "summaries" in path.parts:
            continue
        payload = json.loads(path.read_text())
        config = payload.get("config", {})
        category = infer_category(path, payload)
        row = {
            "path": str(path),
            "category": category,
            "dataset": config.get("dataset"),
            "model": config.get("model"),
            "heuristic": config.get("heuristic"),
            "condition": payload.get("condition")
            or config.get("experiment_condition")
            or category,
            "seed": config.get("seed"),
            "metrics": flatten_metrics(payload),
        }
        rows.append(row)
    return rows


def summarise_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        key = (
            row["category"],
            row["dataset"],
            row["model"],
            row["heuristic"],
            row["condition"],
        )
        for metric, value in row["metrics"].items():
            grouped[key][metric].append(float(value))

    summary_rows: list[dict[str, Any]] = []
    for key, metrics in sorted(grouped.items()):
        category, dataset, model, heuristic, condition = key
        for metric, values in sorted(metrics.items()):
            arr = np.asarray(values, dtype=np.float64)
            summary_rows.append(
                {
                    "category": category,
                    "dataset": dataset,
                    "model": model,
                    "heuristic": heuristic or "",
                    "condition": condition,
                    "metric": metric,
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=0)),
                    "n_seeds": int(arr.size),
                }
            )
    return summary_rows


def paired_cohens_d(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    diff = sample_a - sample_b
    if diff.size < 2:
        return 0.0
    std = float(np.std(diff, ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.mean(diff) / std)


def compute_significance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_by_key: dict[tuple[str, str], dict[int, dict[str, float]]] = defaultdict(
        dict
    )
    comparisons: dict[tuple[str, str, str | None, str], dict[int, dict[str, float]]] = (
        defaultdict(dict)
    )

    for row in rows:
        dataset = row["dataset"]
        model = row["model"]
        seed = row["seed"]
        if dataset is None or model is None or seed is None:
            continue
        if row["category"] == "baseline":
            baseline_by_key[(dataset, model)][int(seed)] = row["metrics"]
        else:
            key = (dataset, model, row["heuristic"], row["condition"])
            comparisons[key][int(seed)] = row["metrics"]

    out: list[dict[str, Any]] = []
    target_metrics = ["heart_mrr", "mrr", "auc"]
    for (dataset, model, heuristic, condition), comp_rows in sorted(
        comparisons.items()
    ):
        baseline_rows = baseline_by_key.get((dataset, model), {})
        shared_seeds = sorted(set(comp_rows) & set(baseline_rows))
        if len(shared_seeds) < 2:
            continue
        for metric in target_metrics:
            comp_values = []
            base_values = []
            for seed in shared_seeds:
                comp_metric = comp_rows[seed].get(metric)
                base_metric = baseline_rows[seed].get(metric)
                if comp_metric is None or base_metric is None:
                    continue
                comp_values.append(comp_metric)
                base_values.append(base_metric)
            if len(comp_values) < 2:
                continue
            comp_arr = np.asarray(comp_values, dtype=np.float64)
            base_arr = np.asarray(base_values, dtype=np.float64)
            t_stat, p_value = ttest_rel(comp_arr, base_arr)
            effect_size = paired_cohens_d(comp_arr, base_arr)
            out.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "heuristic": heuristic or "",
                    "condition": condition,
                    "metric": metric,
                    "t_stat": float(t_stat),
                    "p_value": float(p_value),
                    "effect_size": effect_size,
                    "significant": bool(p_value < 0.05),
                    "n_pairs": len(comp_values),
                }
            )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate experiment result JSON files."
    )
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="results/summaries")
    args = parser.parse_args()

    rows = collect_result_rows(Path(args.results_dir))
    summary_rows = summarise_rows(rows)
    output_dir = Path(args.output_dir)

    for category, filename in SUMMARY_FILES.items():
        subset = [row for row in summary_rows if row["category"] == category]
        write_csv(
            output_dir / filename,
            subset,
            [
                "dataset",
                "model",
                "heuristic",
                "condition",
                "metric",
                "mean",
                "std",
                "n_seeds",
            ],
        )

    significance_rows = compute_significance(rows)
    write_csv(
        output_dir / "significance_tests.csv",
        significance_rows,
        [
            "dataset",
            "model",
            "heuristic",
            "condition",
            "metric",
            "t_stat",
            "p_value",
            "effect_size",
            "significant",
            "n_pairs",
        ],
    )

    print(f"Aggregated {len(rows)} result files into {output_dir}")


if __name__ == "__main__":
    main()
