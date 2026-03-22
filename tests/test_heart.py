from __future__ import annotations

import tempfile

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from experiments.evaluate import run_evaluation
from negative_sampling.heart import (
    HeaRTEvaluator,
    compute_hits_at_k_per_pos,
    compute_per_positive_rank,
)
from utils.data_utils import edge_index_to_edge_set


class _DummyModel:
    def eval(self):
        return self

    def encode(self, x, edge_index):
        del edge_index
        return x

    def decode(self, z, edge_index):
        src = edge_index[0]
        dst = edge_index[1]
        return (z[src] * z[dst]).sum(dim=1)


def _make_data() -> tuple[Data, dict[str, object]]:
    x = torch.tensor(
        [[2.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 2.0]],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 0, 3, 2]],
        dtype=torch.long,
    )
    data = Data(x=x, edge_index=edge_index, num_nodes=4)
    data_dict = {
        "data": data,
        "x": x,
        "train_edge_index": edge_index,
        "train_pos_edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "val_pos_edge_index": torch.tensor([[2], [3]], dtype=torch.long),
        "test_pos_edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "val_neg_edge_index": torch.tensor([[0], [2]], dtype=torch.long),
        "test_neg_edge_index": torch.tensor([[0], [2]], dtype=torch.long),
        "num_nodes": 4,
        "num_features": 2,
    }
    return data, data_dict


def test_compute_per_positive_rank_and_hits() -> None:
    rank = compute_per_positive_rank(1.0, np.zeros(100, dtype=np.float32))
    assert rank == 1
    assert compute_hits_at_k_per_pos(rank, k=10) == 1.0


def test_generate_test_set_avoids_positive_leakage() -> None:
    data, data_dict = _make_data()
    with tempfile.TemporaryDirectory() as tmp:
        np.savez_compressed(
            f"{tmp}/toy_common_neighbors.npz",
            candidates=np.array([[0, 0, 1], [2, 3, 3]], dtype=np.int64),
            scores=np.array([0.9, 0.8, 0.7], dtype=np.float32),
        )
        evaluator = HeaRTEvaluator(
            data=data,
            heuristics=["cn"],
            num_neg_per_pos=2,
            precomputed_dir=tmp,
            seed=0,
        )
        _, hard_negs = evaluator.generate_test_set(data_dict["test_pos_edge_index"])
        all_pos_set = edge_index_to_edge_set(data.edge_index)

        for neg_tensor in hard_negs:
            neg_set = edge_index_to_edge_set(neg_tensor)
            assert all_pos_set.isdisjoint(neg_set)


def test_evaluate_model_returns_heart_metrics() -> None:
    data, data_dict = _make_data()
    with tempfile.TemporaryDirectory() as tmp:
        np.savez_compressed(
            f"{tmp}/toy_common_neighbors.npz",
            candidates=np.array([[0, 0, 1], [2, 3, 3]], dtype=np.int64),
            scores=np.array([0.9, 0.8, 0.7], dtype=np.float32),
        )
        evaluator = HeaRTEvaluator(
            data=data,
            heuristics=["cn"],
            num_neg_per_pos=2,
            precomputed_dir=tmp,
            seed=0,
        )
        model = _DummyModel()
        metrics = evaluator.evaluate_model(model, data_dict, torch.device("cpu"))

        assert set(metrics) == {
            "heart_mrr",
            "heart_hits@10",
            "heart_hits@50",
            "heart_hits@100",
        }
        assert metrics["heart_mrr"] > 0.0


def test_run_evaluation_merges_heart_metrics() -> None:
    data, data_dict = _make_data()
    with tempfile.TemporaryDirectory() as tmp:
        np.savez_compressed(
            f"{tmp}/toy_common_neighbors.npz",
            candidates=np.array([[0, 0, 1], [2, 3, 3]], dtype=np.int64),
            scores=np.array([0.9, 0.8, 0.7], dtype=np.float32),
        )
        evaluator = HeaRTEvaluator(
            data=data,
            heuristics=["cn"],
            num_neg_per_pos=2,
            precomputed_dir=tmp,
            seed=0,
        )
        model = _DummyModel()
        metrics = run_evaluation(
            model, data_dict, torch.device("cpu"), heart_evaluator=evaluator
        )

        assert "auc" in metrics
        assert "heart_mrr" in metrics


def test_missing_precomputed_file_raises() -> None:
    data, _ = _make_data()
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError, match="Run scripts/precompute_scores.py"):
            HeaRTEvaluator(data=data, heuristics=["cn"], precomputed_dir=tmp)
