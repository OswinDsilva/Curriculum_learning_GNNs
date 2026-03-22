from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(results_dir: str | Path) -> pd.DataFrame:
    """Load experiment JSON outputs into a flat DataFrame for plotting."""
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(results_dir).rglob("*.json")):
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
            "phase_history": payload.get("phase_history")
            or payload.get("phase_summary", {}).get("phase_history", []),
        }
        for section in ("standard", "heart"):
            for key, value in (payload.get(section, {}) or {}).items():
                row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def load_epoch_curves(results_dir: str | Path) -> pd.DataFrame:
    """Load per-epoch CSV logs into a DataFrame for learning-curve plots."""
    frames: list[pd.DataFrame] = []
    for path in sorted(Path(results_dir).rglob("*_epochs.csv")):
        frame = pd.read_csv(path)
        stem = path.stem.replace("_epochs", "")
        parts = stem.split("_")
        frame["run_name"] = stem
        frame["dataset"] = parts[0] if len(parts) > 0 else None
        frame["model"] = parts[1] if len(parts) > 1 else None
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _save_current_figure(base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(base_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def _placeholder_plot(title: str, message: str, save_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.title(title)
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    plt.axis("off")
    _save_current_figure(save_path)


def plot_learning_curves(
    curves: pd.DataFrame, results_df: pd.DataFrame, figures_dir: Path
) -> None:
    """Plot baseline vs curriculum validation AUC over epochs where CSV logs exist."""
    if curves.empty or "auc" not in curves.columns:
        _placeholder_plot(
            "Learning Curves",
            "Insufficient epoch CSV data available.",
            figures_dir / "learning_curves_cora_gcn",
        )
        return

    for (dataset, model), group in curves.groupby(["dataset", "model"], dropna=False):
        plt.figure(figsize=(8, 5))
        if "phase" in group.columns:
            pass
        for run_name, run_frame in group.groupby("run_name"):
            label = (
                "curriculum"
                if "common_neighbors" in run_name or "abl-" in run_name
                else "baseline"
            )
            plt.plot(run_frame["epoch"], run_frame["auc"], alpha=0.7, label=label)

        for _, row in results_df[
            (results_df["dataset"] == dataset) & (results_df["model"] == model)
        ].iterrows():
            for event in row.get("phase_history", []) or []:
                epoch = event.get("epoch") if isinstance(event, dict) else None
                if epoch and epoch > 0:
                    plt.axvline(epoch, color="gray", linestyle="--", alpha=0.2)
        plt.xlabel("Epoch")
        plt.ylabel("Validation AUC")
        plt.title(f"Learning Curves: {dataset} / {model}")
        handles, labels = plt.gca().get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        plt.legend(uniq.values(), uniq.keys())
        _save_current_figure(figures_dir / f"learning_curves_{dataset}_{model}")


def plot_performance_comparison(df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot grouped HeaRT MRR comparisons by dataset/model/condition."""
    if df.empty or "heart_mrr" not in df.columns:
        _placeholder_plot(
            "Performance Comparison",
            "No HeaRT metrics available.",
            figures_dir / "performance_comparison_cora",
        )
        return
    for dataset, group in df.groupby("dataset", dropna=False):
        pivot = (
            group.groupby(["model", "condition"], dropna=False)["heart_mrr"]
            .mean()
            .unstack(fill_value=np.nan)
        )
        if pivot.empty:
            _placeholder_plot(
                f"Performance Comparison: {dataset}",
                "No comparable rows available.",
                figures_dir / f"performance_comparison_{dataset}",
            )
            continue
        ax = pivot.plot(kind="bar", figsize=(9, 5))
        ax.set_ylabel("HeaRT MRR")
        ax.set_title(f"Performance Comparison: {dataset}")
        plt.xticks(rotation=0)
        _save_current_figure(figures_dir / f"performance_comparison_{dataset}")


def plot_heart_gap(df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot standard MRR vs HeaRT MRR."""
    if df.empty or "mrr" not in df.columns or "heart_mrr" not in df.columns:
        _placeholder_plot(
            "HeaRT Gap Analysis",
            "Need both standard and HeaRT MRR columns.",
            figures_dir / "heart_gap_analysis",
        )
        return
    plt.figure(figsize=(6, 6))
    for dataset, group in df.groupby("dataset", dropna=False):
        plt.scatter(group["mrr"], group["heart_mrr"], label=dataset)
    limit = max(float(df["mrr"].max()), float(df["heart_mrr"].max()), 1e-6)
    plt.plot([0, limit], [0, limit], linestyle="--", color="gray")
    plt.xlabel("Standard MRR")
    plt.ylabel("HeaRT MRR")
    plt.title("HeaRT Gap Analysis")
    plt.legend()
    _save_current_figure(figures_dir / "heart_gap_analysis")


def plot_heuristic_comparison(df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot heuristic comparison across datasets for each model."""
    subset = df[df["heuristic"].notna() & (df["heuristic"] != "")]
    if subset.empty or "heart_mrr" not in subset.columns:
        _placeholder_plot(
            "Heuristic Comparison",
            "No heuristic-tagged HeaRT runs available.",
            figures_dir / "heuristic_comparison",
        )
        return
    models = sorted(subset["model"].dropna().unique())
    fig, axes = plt.subplots(
        1, max(len(models), 1), figsize=(6 * max(len(models), 1), 4), squeeze=False
    )
    for ax, model in zip(axes[0], models):
        model_df = subset[subset["model"] == model]
        pivot = model_df.groupby(["dataset", "heuristic"])["heart_mrr"].mean().unstack()
        pivot.plot(ax=ax, marker="o")
        ax.set_title(f"Heuristics: {model}")
        ax.set_ylabel("HeaRT MRR")
    _save_current_figure(figures_dir / "heuristic_comparison")


def plot_phase_transitions(df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot distribution of curriculum phase transition epochs."""
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        for event in row.get("phase_history", []) or []:
            if isinstance(event, dict) and event.get("epoch", 0) > 0:
                records.append(
                    {"phase_name": event.get("phase_name"), "epoch": event.get("epoch")}
                )
    phase_df = pd.DataFrame(records)
    if phase_df.empty:
        _placeholder_plot(
            "Phase Transitions",
            "No phase transition events available.",
            figures_dir / "phase_transitions",
        )
        return
    plt.figure(figsize=(8, 4))
    phase_df.boxplot(column="epoch", by="phase_name")
    plt.suptitle("")
    plt.title("Phase Transition Epochs")
    plt.ylabel("Epoch")
    _save_current_figure(figures_dir / "phase_transitions")


def plot_competence_progression(curves: pd.DataFrame, figures_dir: Path) -> None:
    """Plot curriculum competence progression proxy using validation AUC curves."""
    if curves.empty or "auc" not in curves.columns:
        _placeholder_plot(
            "Competence Progression",
            "No epoch AUC logs available.",
            figures_dir / "competence_progression",
        )
        return
    datasets = sorted(curves["dataset"].dropna().unique())
    fig, axes = plt.subplots(
        len(datasets), 1, figsize=(8, 4 * max(len(datasets), 1)), squeeze=False
    )
    for ax, dataset in zip(axes[:, 0], datasets):
        dataset_df = curves[curves["dataset"] == dataset]
        for run_name, run_frame in dataset_df.groupby("run_name"):
            ax.plot(run_frame["epoch"], run_frame["auc"], alpha=0.6, label=run_name)
        ax.set_title(f"Competence Progression: {dataset}")
        ax.set_ylabel("Validation AUC")
    axes[-1, 0].set_xlabel("Epoch")
    _save_current_figure(figures_dir / "competence_progression")


def plot_ablation_study(df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot HeaRT MRR across ablation conditions."""
    ablations = df[df["condition"].astype(str).str.startswith("abl-")]
    if ablations.empty or "heart_mrr" not in ablations.columns:
        _placeholder_plot(
            "Ablation Study",
            "No ablation runs with HeaRT metrics available.",
            figures_dir / "ablation_study",
        )
        return
    summary = ablations.groupby("condition")["heart_mrr"].mean().sort_values()
    plt.figure(figsize=(8, 5))
    colors = ["#cc5533" if idx == "abl-5" else "#4c78a8" for idx in summary.index]
    plt.barh(summary.index, summary.values, color=colors)
    plt.xlabel("HeaRT MRR")
    plt.title("Ablation Study")
    _save_current_figure(figures_dir / "ablation_study")


def main() -> None:
    """Generate Phase 7 figures from current experiment outputs."""
    parser = argparse.ArgumentParser(
        description="Generate analysis figures from result files."
    )
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--figures_dir", type=str, default="results/figures")
    args = parser.parse_args()

    results_df = load_results(args.results_dir)
    curves_df = load_epoch_curves(args.results_dir)
    figures_dir = Path(args.figures_dir)

    plot_learning_curves(curves_df, results_df, figures_dir)
    plot_performance_comparison(results_df, figures_dir)
    plot_heart_gap(results_df, figures_dir)
    plot_heuristic_comparison(results_df, figures_dir)
    plot_phase_transitions(results_df, figures_dir)
    plot_competence_progression(curves_df, figures_dir)
    plot_ablation_study(results_df, figures_dir)
    print(f"Generated figures in {figures_dir}")


if __name__ == "__main__":
    main()
