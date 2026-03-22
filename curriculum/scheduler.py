from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from curriculum.competence import CompetenceMeter


@dataclass(frozen=True)
class CurriculumPhase:
    difficulty_ratios: list[float]
    threshold: float | None
    name: str = ""


DEFAULT_PHASES = [
    CurriculumPhase([1.0, 0.0, 0.0], threshold=0.75, name="easy_only"),
    CurriculumPhase([0.7, 0.3, 0.0], threshold=0.85, name="easy_medium"),
    CurriculumPhase([0.3, 0.4, 0.3], threshold=0.90, name="mixed"),
    CurriculumPhase([0.0, 0.3, 0.7], threshold=None, name="hard_focus"),
]


class CurriculumScheduler:
    def __init__(
        self,
        phases: list[CurriculumPhase] | None = None,
        adaptive: bool = True,
        fixed_phase_epochs: int = 75,
        competence_window: int = 5,
    ) -> None:
        self.phases = list(phases) if phases is not None else list(DEFAULT_PHASES)
        if not self.phases:
            raise ValueError("phases must not be empty.")
        if fixed_phase_epochs <= 0:
            raise ValueError("fixed_phase_epochs must be positive.")

        self.adaptive = adaptive
        self.fixed_phase_epochs = fixed_phase_epochs
        self.current_phase_idx = 0
        self.phase_changed = False
        self.phase_history: list[tuple[int, str]] = [(0, self.current_phase.name)]
        self.competence_meter = CompetenceMeter(window_size=competence_window)

    def step(self, metric_value: float, epoch: int) -> None:
        self.phase_changed = False
        self.competence_meter.update(metric_value)

        if self.adaptive:
            if self.should_advance():
                self.advance_phase(epoch)
            return

        if self.is_final_phase:
            return
        if epoch > 0 and epoch % self.fixed_phase_epochs == 0:
            self.advance_phase(epoch)

    def should_advance(self) -> bool:
        if self.is_final_phase:
            return False
        threshold = self.current_phase.threshold
        if threshold is None:
            return False
        return self.competence_meter.is_threshold_reached(threshold)

    def advance_phase(self, epoch: int) -> None:
        if self.is_final_phase:
            return
        self.current_phase_idx += 1
        self.phase_changed = True
        self.phase_history.append((epoch, self.current_phase.name))
        self.competence_meter.reset()
        print(
            f"[Epoch {epoch}] Advancing to Phase {self.current_phase_idx}: {self.current_phase.name}"
        )

    def get_current_difficulty_ratios(self) -> list[float]:
        return list(self.current_phase.difficulty_ratios)

    def get_phase_summary(self) -> dict[str, Any]:
        return {
            "current_phase": self.current_phase_idx,
            "phase_name": self.current_phase.name,
            "phase_history": [
                {"epoch": epoch, "phase_name": name}
                for epoch, name in self.phase_history
            ],
            "competence": self.competence_meter.get_competence(),
            "adaptive": self.adaptive,
        }

    @property
    def current_phase(self) -> CurriculumPhase:
        return self.phases[self.current_phase_idx]

    @property
    def is_final_phase(self) -> bool:
        return self.current_phase_idx >= len(self.phases) - 1
