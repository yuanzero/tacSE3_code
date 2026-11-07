"""
Contact-region centroid in the sensor plane, and plane displacement from its motion.

Centroids use the same metric grid as ``estimate_rotation_from_shear`` (``pixel_xy_grid``).
Displacement is either consecutive frame-to-frame or sliding-window (front vs back halves),
matching the spirit of the realtime sensor loop rotation smoothing.

Two-sensor translation fusion matches ``sensor_loop.simple_sensor_loop_cli``:
average on X/Z, difference-based on Y.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .core.estimator import pixel_xy_grid


def contact_centroid_xy_m(mask: Optional[np.ndarray], ppmm: float) -> Optional[Tuple[float, float]]:
    """
    Centroid of the contact mask in sensor plane coordinates (meters).

    Same frame as shear/rotation estimation: origin at image center, X/Y from ``pixel_xy_grid``.

    Returns:
        ``(cx, cy)`` in meters, or ``None`` if ``mask`` is empty or has no True pixels.
    """
    if mask is None:
        return None
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not np.any(m):
        return None
    H, W = m.shape
    X, Y = pixel_xy_grid((H, W), float(ppmm))
    yy, xx = np.nonzero(m)
    if yy.size == 0:
        return None
    return float(np.mean(X[yy, xx])), float(np.mean(Y[yy, xx]))


def fuse_plane_translation_deltas(sensor_deltas: Sequence[Union[np.ndarray, Sequence[float]]]) -> np.ndarray:
    """
    Fuse per-sensor plane deltas into one ``[dx, dy, dz]`` (meters).

    - 0 sensors: zeros
    - 1 sensor: pass through (dz = 0 if only xy given)
    - 2 sensors: same rule as the main sensor loop — X and Z use ``(t0 + t1) / 2``; Y is
      replaced by ``(t0_y - t1_y) / 2`` (third component is usually 0 for plane motion).
    - 3+ sensors: elementwise mean
    """
    outs: List[np.ndarray] = []
    for d in sensor_deltas:
        a = np.asarray(d, dtype=float).ravel()
        if a.size < 3:
            a = np.pad(a, (0, max(0, 3 - a.size)))
        outs.append(a[:3].astype(float))

    n = len(outs)
    if n == 0:
        return np.zeros(3, dtype=float)
    if n == 1:
        return outs[0]
    if n == 2:
        t0, t1 = outs[0], outs[1]
        out = (t0 + t1) / 2.0
        out[1] = (t0[1] - t1[1]) / 2.0
        return out
    return np.mean(np.stack(outs, axis=0), axis=0)


class ContactCenterTranslationState:
    """
    Stateful estimator: per-frame fused ``[dx, dy, 0]`` from contact mask centroids.

    Parameters
    ----------
    num_sensors
        Number of sensors; ``update`` must receive this many masks each call.
    mode
        ``\"consecutive\"`` — displacement = current centroid − previous centroid (per sensor).
        ``\"window\"`` — when the ring buffer has ``2 * smooth_window`` samples, displacement =
        mean(last ``smooth_window`` centroids) − mean(first ``smooth_window`` centroids); until
        then, falls back to the same consecutive rule so motion is not stuck at zero.
    smooth_window
        Half-window length for ``mode=\"window\"`` (full buffer length = ``2 * smooth_window``).
    """

    def __init__(
        self,
        num_sensors: int,
        *,
        mode: str = "consecutive",
        smooth_window: int = 2,
    ) -> None:
        if mode not in ("consecutive", "window"):
            raise ValueError('mode must be \"consecutive\" or \"window\"')
        self.num_sensors = int(num_sensors)
        if self.num_sensors < 1:
            raise ValueError("num_sensors must be >= 1")
        self.mode = mode
        self.smooth_window = max(1, int(smooth_window))
        self.buffer_size = 2 * self.smooth_window
        self._prev_xy: List[Optional[Tuple[float, float]]] = [None] * self.num_sensors
        self._buffers: List[deque] = [deque(maxlen=self.buffer_size) for _ in range(self.num_sensors)]

    def reset(self) -> None:
        """Clear history (after lost contact or session restart)."""
        self._prev_xy = [None] * self.num_sensors
        for b in self._buffers:
            b.clear()

    def _delta_one_sensor(self, idx: int, c: Optional[Tuple[float, float]]) -> Tuple[float, float]:
        if c is None:
            self._prev_xy[idx] = None
            self._buffers[idx].clear()
            return 0.0, 0.0

        px, py = c
        if self.mode == "consecutive":
            pr = self._prev_xy[idx]
            self._prev_xy[idx] = c
            if pr is None:
                return 0.0, 0.0
            return px - pr[0], py - pr[1]

        buf = self._buffers[idx]
        buf.append(c)
        if len(buf) >= self.buffer_size:
            first = list(buf)[: self.smooth_window]
            last = list(buf)[-self.smooth_window :]
            mx0 = float(np.mean([p[0] for p in first]))
            my0 = float(np.mean([p[1] for p in first]))
            mx1 = float(np.mean([p[0] for p in last]))
            my1 = float(np.mean([p[1] for p in last]))
            self._prev_xy[idx] = c
            return mx1 - mx0, my1 - my0

        pr = self._prev_xy[idx]
        self._prev_xy[idx] = c
        if pr is None:
            return 0.0, 0.0
        return px - pr[0], py - pr[1]

    def update(self, masks: Sequence[np.ndarray], ppmm: float) -> np.ndarray:
        """
        Compute fused plane displacement increment ``[dx, dy, dz]`` (meters, dz=0).

        Parameters
        ----------
        masks
            One boolean ``(H, W)`` contact mask per sensor (same order every call).
        ppmm
            Pixels per millimeter for ``pixel_xy_grid``.
        """
        if len(masks) != self.num_sensors:
            raise ValueError(f"expected {self.num_sensors} masks, got {len(masks)}")
        ppmm_f = float(ppmm)
        per_sensor: List[np.ndarray] = []
        for i, mask in enumerate(masks):
            c = contact_centroid_xy_m(mask, ppmm_f)
            dx, dy = self._delta_one_sensor(i, c)
            per_sensor.append(np.array([dx, dy, 0.0], dtype=float))
        return fuse_plane_translation_deltas(per_sensor)


def fused_contact_center_delta_m(
    masks: Sequence[np.ndarray],
    ppmm: float,
    state: ContactCenterTranslationState,
) -> np.ndarray:
    """Convenience wrapper: ``return state.update(masks, ppmm)``."""
    return state.update(masks, ppmm)


__all__ = [
    "contact_centroid_xy_m",
    "fuse_plane_translation_deltas",
    "ContactCenterTranslationState",
    "fused_contact_center_delta_m",
]
