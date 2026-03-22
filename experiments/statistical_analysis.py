from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from scipy.stats import ttest_rel


def load_result_records(results_dir: str | Path) -> pd.DataFrame:
    """Load all experiment JSON files under a results directory into a DataFrame."""
    root = Path(results_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        if "summaries" in path.parts:
            continue
        payload = json.loads(path.read_text())
        config = payload.get("config", {})
        row: dict[str, Any] = {
            "path": str(path),
            "dataset": config.get("dataset"),
            "model": config.get("model"),
            "heuristic": config.get("heuristic"),
            "condition": payload.get("condition")
            or config.get("experiment_condition")
            or "curriculum",
            "seed": config.get("seed"),
        }
        for section in ("standard", "heart"):
            metrics = payload.get(section, {}) or {}
            for key, value in metrics.items():
                row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def cohens_d(a: list[float], b: list[float]) -> float:
    """Compute Cohen's d using pooled standard deviation."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.size < 2 or arr_b.size < 2:
        return 0.0
    diff = float(arr_a.mean() - arr_b.mean())
    pooled_std = np.sqrt(
        (np.std(arr_a, ddof=1) ** 2 + np.std(arr_b, ddof=1) ** 2) / 2.0
    )
    if pooled_std == 0.0:
        return 0.0
    return diff / float(pooled_std)


def confidence_interval_95(values: list[float]) -> tuple[float, float]:
    """Compute the 95% confidence interval for a sample mean."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (0.0, 0.0)
    if arr.size == 1:
        value = float(arr[0])
        return (value, value)
    mean = float(arr.mean())
    se = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    margin = float(t_dist.ppf(0.975, df=arr.size - 1) * se)
    return (mean - margin, mean + margin)


def build_full_significance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the curriculum-vs-baseline significance table used in the report."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "heuristic",
                "metric",
                "baseline_mean",
                "baseline_std",
                "curriculum_mean",
                "curriculum_std",
                "improvement_pct",
                "t_stat",
                "p_value",
                "cohens_d",
                "significant",
            ]
        )

    baseline = df[df["condition"] == "baseline"]
    curriculum = df[~df["condition"].isin(["curriculum", "fixed_epoch"])]
    # Include main curriculum condition explicitly while excluding ablations.
    curriculum = pd.concat(
        [curriculum, df[df["condition"] == "curriculum"]], ignore_index=True
    )
    dedup_cols = [
        col
        for col in ["path", "dataset", "model", "heuristic", "condition", "seed"]
        if col in curriculum.columns
    ]
    if dedup_cols:
        curriculum = curriculum.drop_duplicates(subset=dedup_cols)

    rows: list[dict[str, Any]] = []
    for (dataset, model, heuristic), cur_group in curriculum.groupby(
        ["dataset", "model", "heuristic"], dropna=False
    ):
        base_group = baseline[
            (baseline["dataset"] == dataset) & (baseline["model"] == model)
        ]
        if base_group.empty:
            continue
        merged = pd.merge(
            cur_group,
            base_group,
            on=["dataset", "model", "seed"],
            suffixes=("_cur", "_base"),
        )
        for metric in ["heart_mrr", "heart_hits@10", "heart_hits@50"]:
            cur_col = f"{metric}_cur"
            base_col = f"{metric}_base"
            if cur_col not in merged or base_col not in merged:
                continue
            cur_vals = merged[cur_col].dropna().astype(float).tolist()
            base_vals = merged[base_col].dropna().astype(float).tolist()
            if not cur_vals or len(cur_vals) != len(base_vals):
                continue
            if len(cur_vals) >= 2:
                t_stat, p_value = ttest_rel(cur_vals, base_vals)
            else:
                t_stat, p_value = 0.0, 1.0
            base_mean = float(np.mean(base_vals))
            cur_mean = float(np.mean(cur_vals))
            improvement_pct = (
                0.0 if base_mean == 0.0 else 100.0 * (cur_mean - base_mean) / base_mean
            )
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "heuristic": heuristic or "",
                    "metric": metric,
                    "baseline_mean": base_mean,
                    "baseline_std": float(np.std(base_vals, ddof=0)),
                    "curriculum_mean": cur_mean,
                    "curriculum_std": float(np.std(cur_vals, ddof=0)),
                    "improvement_pct": improvement_pct,
                    "t_stat": float(t_stat),
                    "p_value": float(p_value),
                    "cohens_d": float(cohens_d(cur_vals, base_vals)),
                    "significant": bool(p_value < 0.05),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Generate the full significance table CSV from experiment JSON files."""
    parser = argparse.ArgumentParser(
        description="Run statistical analysis on experiment results."
    )
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/summaries/full_significance_table.csv",
    )
    args = parser.parse_args()

    df = load_result_records(args.results_dir)
    table = build_full_significance_table(df)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(f"Wrote {len(table)} significance rows to {output_path}")


if __name__ == "__main__":
    main()
