from __future__ import annotations

from collections import deque
from typing import Literal


class CompetenceMeter:
    def __init__(
        self,
        metric: str = "val_auc",
        window_size: int = 5,
        smoothing: Literal["moving_average", "ema"] = "moving_average",
        ema_alpha: float = 0.5,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if smoothing not in {"moving_average", "ema"}:
            raise ValueError("smoothing must be 'moving_average' or 'ema'.")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1].")

        self.metric = metric
        self.window_size = window_size
        self.smoothing = smoothing
        self.ema_alpha = ema_alpha
        self._history: deque[float] = deque(maxlen=window_size)
        self._ema_value: float | None = None

    def update(self, metric_value: float) -> None:
        value = float(metric_value)
        self._history.append(value)
        if self._ema_value is None:
            self._ema_value = value
        else:
            self._ema_value = (
                self.ema_alpha * value + (1.0 - self.ema_alpha) * self._ema_value
            )

    def get_competence(self) -> float:
        if not self._history:
            return 0.0
        if self.smoothing == "ema":
            return float(self._ema_value if self._ema_value is not None else 0.0)
        return float(sum(self._history) / len(self._history))

    def is_threshold_reached(self, threshold: float) -> bool:
        return self.get_competence() >= threshold

    def reset(self) -> None:
        self._history.clear()
        self._ema_value = None

    @property
    def history(self) -> list[float]:
        return list(self._history)
