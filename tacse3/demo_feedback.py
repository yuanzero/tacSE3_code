"""Display-only stabilization. Never changes estimator results or robot actions."""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class FeedbackConfig:
    """Small, explicit visualization parameters, independent of the estimator."""

    smoothing_ms: float = 120.0
    min_contact_pixels: int = 100
    activation_ms: float = 60.0
    max_gap_s: float = 0.3

    def __post_init__(self):
        for name in ("smoothing_ms", "activation_ms"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.min_contact_pixels < 3:
            raise ValueError("min_contact_pixels must be at least 3")
        if not math.isfinite(self.max_gap_s) or self.max_gap_s <= 0:
            raise ValueError("max_gap_s must be finite and positive")


class DisplayFeedback:
    """Three-frame median and time-aware EMA, reset immediately on contact loss.

    The result is a display vector, not an integrated pose. A new contact starts
    a new filter segment; no stale pose is held through gaps. Raw input is never
    mutated. Setting smoothing_ms=0 disables both median and EMA.
    """

    def __init__(self, config=None):
        self.config = config or FeedbackConfig()
        self.reset()

    def reset(self):
        self.window = deque(maxlen=3)
        self.started = None
        self.previous_time = None
        self.filtered = None

    def update(self, result):
        timestamp = float(result["time"])
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        raw = result.get("rotation_vector_rad")
        state = "tracking"
        if raw is None:
            state = "light_response" if result.get("response_pixels", 0) else "idle"
        elif result.get("rank", 0) < 5:
            state = "weak_fit"
        elif result.get("contact_pixels", 0) < self.config.min_contact_pixels:
            state = "small_contact"
        else:
            raw = np.asarray(raw, dtype=float)
            if raw.shape != (3,) or not np.isfinite(raw).all():
                state = "invalid_estimate"
        if state != "tracking":
            self.reset()
            return dict(state=state, rotation_vector_rad=None)
        if (
            self.previous_time is not None
            and not 0 < timestamp - self.previous_time <= self.config.max_gap_s
        ):
            self.reset()
        if self.started is None:
            self.started = timestamp
        dt = 0.0 if self.previous_time is None else timestamp - self.previous_time
        self.previous_time = timestamp
        self.window.append(raw.copy())
        if self.config.smoothing_ms == 0:
            self.filtered = raw.copy()
        else:
            target = np.median(np.stack(self.window), axis=0)
            alpha = -math.expm1(-dt / (self.config.smoothing_ms / 1000.0))
            self.filtered = (
                target.copy()
                if self.filtered is None
                else self.filtered + alpha * (target - self.filtered)
            )
        if timestamp - self.started < self.config.activation_ms / 1000.0:
            return dict(state="settling", rotation_vector_rad=None)
        return dict(state="tracking", rotation_vector_rad=self.filtered.tolist())
