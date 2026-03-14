from __future__ import annotations

import tempfile

import numpy as np
import pytest
import torch

from curriculum.competence import CompetenceMeter
from curriculum.scheduler import CurriculumPhase, CurriculumScheduler
from experiments.train_curriculum import (
    load_precomputed_candidates,
    resolve_heuristic_name,
)


def test_competence_meter_window_and_threshold() -> None:
    meter = CompetenceMeter(window_size=3)
    meter.update(0.80)
    meter.update(0.82)
    meter.update(0.84)

    assert abs(meter.get_competence() - 0.82) < 1e-6
    assert meter.is_threshold_reached(0.80)
    assert not meter.is_threshold_reached(0.85)


def test_competence_meter_drops_oldest() -> None:
    meter = CompetenceMeter(window_size=3)
    for value in [0.70, 0.80, 0.90, 1.00]:
        meter.update(value)

    assert meter.history == [0.80, 0.90, 1.00]
    assert meter.get_competence() == pytest.approx(0.90)


def test_competence_meter_reset() -> None:
    meter = CompetenceMeter(window_size=2)
    meter.update(0.75)
    meter.reset()

    assert meter.history == []
    assert meter.get_competence() == 0.0


def test_scheduler_adaptive_advances() -> None:
    sched = CurriculumScheduler(adaptive=True, competence_window=5)

    assert sched.current_phase_idx == 0
    assert sched.get_current_difficulty_ratios() == [1.0, 0.0, 0.0]

    sched.step(0.76, epoch=1)

    assert sched.current_phase_idx == 1
    assert sched.phase_changed
    assert sched.phase_history[-1] == (1, "easy_medium")


def test_scheduler_fixed_advances() -> None:
    sched = CurriculumScheduler(adaptive=False, fixed_phase_epochs=2)

    sched.step(0.1, epoch=1)
    assert sched.current_phase_idx == 0

    sched.step(0.1, epoch=2)
    assert sched.current_phase_idx == 1
    assert sched.phase_changed


def test_scheduler_phase_summary_shape() -> None:
    sched = CurriculumScheduler(adaptive=True)
    summary = sched.get_phase_summary()

    assert summary["current_phase"] == 0
    assert summary["phase_name"] == "easy_only"
    assert isinstance(summary["phase_history"], list)
    assert summary["phase_history"][0]["phase_name"] == "easy_only"


def test_scheduler_resets_competence_on_advance() -> None:
    phases = [
        CurriculumPhase([1.0, 0.0, 0.0], threshold=0.6, name="p0"),
        CurriculumPhase([0.0, 1.0, 0.0], threshold=None, name="p1"),
    ]
    sched = CurriculumScheduler(phases=phases, adaptive=True, competence_window=2)
    sched.step(0.7, epoch=1)

    assert sched.current_phase_idx == 1
    assert sched.competence_meter.get_competence() == 0.0


def test_resolve_heuristic_name_aliases() -> None:
    assert resolve_heuristic_name("cn") == "common_neighbors"
    assert resolve_heuristic_name("aa") == "adamic_adar"
    assert resolve_heuristic_name("ra") == "resource_allocation"


def test_resolve_heuristic_name_invalid() -> None:
    with pytest.raises(ValueError, match="Unknown heuristic"):
        resolve_heuristic_name("bogus")


def test_load_precomputed_candidates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/cora_common_neighbors.npz"
        np.savez_compressed(
            path,
            candidates=np.array([[0, 1], [2, 3]], dtype=np.int64),
            scores=np.array([0.1, 0.2], dtype=np.float32),
        )

        candidates, scores = load_precomputed_candidates("cora", "cn", tmp)

        assert torch.equal(candidates, torch.tensor([[0, 1], [2, 3]]))
        assert scores.tolist() == pytest.approx([0.1, 0.2])


def test_load_precomputed_candidates_missing_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError, match="Run scripts/precompute_scores.py"):
            load_precomputed_candidates("cora", "cn", tmp)