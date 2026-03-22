from __future__ import annotations

import json
import tempfile
from pathlib import Path

from experiments.statistical_analysis import (
    build_full_significance_table,
    cohens_d,
    confidence_interval_95,
    load_result_records,
)


def test_cohens_d_zero_when_identical() -> None:
    assert cohens_d([1.0, 1.0], [1.0, 1.0]) == 0.0


def test_confidence_interval_single_value() -> None:
    assert confidence_interval_95([0.5]) == (0.5, 0.5)


def test_load_result_records() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "baseline" / "cora_gcn_seed0.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "config": {"dataset": "cora", "model": "gcn", "seed": 0},
                    "condition": "baseline",
                    "standard": {"mrr": 0.2},
                    "heart": {"heart_mrr": 0.1},
                }
            )
        )
        df = load_result_records(tmp)
        assert len(df) == 1
        assert float(df.iloc[0]["heart_mrr"]) == 0.1


def test_build_full_significance_table() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        baseline_dir = root / "baseline"
        curriculum_dir = root / "curriculum"
        baseline_dir.mkdir()
        curriculum_dir.mkdir()
        for seed, base_val, cur_val in [(0, 0.10, 0.20), (1, 0.12, 0.25)]:
            (baseline_dir / f"cora_gcn_seed{seed}.json").write_text(
                json.dumps(
                    {
                        "config": {
                            "dataset": "cora",
                            "model": "gcn",
                            "seed": seed,
                            "experiment_condition": "baseline",
                        },
                        "condition": "baseline",
                        "heart": {
                            "heart_mrr": base_val,
                            "heart_hits@10": base_val,
                            "heart_hits@50": base_val,
                        },
                    }
                )
            )
            (curriculum_dir / f"cora_gcn_common_neighbors_seed{seed}.json").write_text(
                json.dumps(
                    {
                        "config": {
                            "dataset": "cora",
                            "model": "gcn",
                            "heuristic": "cn",
                            "seed": seed,
                        },
                        "condition": "curriculum",
                        "heart": {
                            "heart_mrr": cur_val,
                            "heart_hits@10": cur_val,
                            "heart_hits@50": cur_val,
                        },
                    }
                )
            )
        df = load_result_records(root)
        table = build_full_significance_table(df)
        assert not table.empty
        assert set(table["metric"]) >= {"heart_mrr", "heart_hits@10", "heart_hits@50"}
