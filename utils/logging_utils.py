from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """Logs per-epoch metrics to TensorBoard and an in-memory buffer for CSV export."""

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_tensorboard: bool = True,
    ) -> None:
        self._records: list[Dict[str, Any]] = []
        self._writer: Optional[SummaryWriter] = None

        if use_tensorboard:
            tb_dir = Path(log_dir) / experiment_name
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(tb_dir))

    def log_epoch(
        self,
        epoch: int,
        loss: float,
        metrics: Dict[str, float],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record loss and metrics for one epoch."""
        record: Dict[str, Any] = {"epoch": epoch, "loss": loss, **metrics}
        if extra:
            record.update(extra)
        self._records.append(record)

        if self._writer is not None:
            self._writer.add_scalar("train/loss", loss, epoch)
            for key, value in metrics.items():
                self._writer.add_scalar(f"val/{key}", value, epoch)

    def save_csv(self, path: str) -> None:
        """Dump all logged epoch data to a CSV file."""
        if not self._records:
            return
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self._records[0].keys())
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._records)

    def save_results_json(self, path: str, final_metrics: Dict[str, Any]) -> None:
        """Save hyperparams and final test metrics as a JSON file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(final_metrics, f, indent=2)

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()


class CheckpointManager:
    """Saves and loads model checkpoints, tracking the best metric value seen."""

    def __init__(
        self,
        checkpoint_dir: str,
        experiment_name: str,
        monitor: str = "val_auc",
    ) -> None:
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._name = experiment_name
        self._monitor = monitor
        self._best_value: float = -float("inf")
        self._best_path: Optional[Path] = None

    def save(
        self,
        model: nn.Module,
        epoch: int,
        metric_value: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Save checkpoint if metric_value improves on the best seen so far."""
        if metric_value <= self._best_value:
            return False

        self._best_value = metric_value
        path = self._dir / f"{self._name}_best.pt"
        payload: Dict[str, Any] = {
            "epoch": epoch,
            self._monitor: metric_value,
            "model_state_dict": model.state_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        self._best_path = path
        return True

    def load_best(self, model: nn.Module) -> nn.Module:
        """Load the best saved weights into model in-place and return it."""
        if self._best_path is None or not self._best_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {self._best_path}. "
                "Call save() at least once before load_best()."
            )
        payload = torch.load(self._best_path, map_location="cpu")
        model.load_state_dict(payload["model_state_dict"])
        return model

    @property
    def best_value(self) -> float:
        return self._best_value
