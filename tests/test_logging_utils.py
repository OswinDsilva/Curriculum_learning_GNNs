from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from utils.logging_utils import CheckpointManager, ExperimentLogger


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)


def test_experiment_logger_save_csv() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        logger = ExperimentLogger(tmp, "test_exp", use_tensorboard=False)
        logger.log_epoch(1, 0.9, {"auc": 0.75, "ap": 0.80})
        logger.log_epoch(2, 0.8, {"auc": 0.82, "ap": 0.85})
        csv_path = f"{tmp}/out.csv"
        logger.save_csv(csv_path)
        logger.close()

        content = Path(csv_path).read_text()
        assert "epoch" in content
        assert "auc" in content
        assert "0.75" in content


def test_experiment_logger_save_json() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        logger = ExperimentLogger(tmp, "test_exp", use_tensorboard=False)
        json_path = f"{tmp}/result.json"
        logger.save_results_json(json_path, {"auc": 0.88, "config": {"lr": 0.01}})
        logger.close()

        data = json.loads(Path(json_path).read_text())
        assert data["auc"] == 0.88


def test_checkpoint_manager_save_load() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        model_a = _TinyModel()
        manager = CheckpointManager(tmp, "exp1")

        saved = manager.save(model_a, epoch=10, metric_value=0.85)
        assert saved
        assert manager.best_value == 0.85

        # Lower metric should not overwrite
        not_saved = manager.save(model_a, epoch=15, metric_value=0.80)
        assert not not_saved

        model_b = _TinyModel()
        # Scramble model_b weights so it differs from model_a
        with torch.no_grad():
            model_b.linear.weight.fill_(99.0)

        manager.load_best(model_b)
        w_a = model_a.linear.weight.detach()
        w_b = model_b.linear.weight.detach()
        assert torch.allclose(w_a, w_b), "Loaded weights should match saved model"
