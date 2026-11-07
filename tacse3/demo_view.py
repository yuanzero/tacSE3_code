"""Rendering helpers for the ball demo; no sensor SDK imports."""

import numpy as np
from .core.units import DMV1_CODE_SHEAR_TO_M


def response_metrics(depth, shear):
    """Unthresholded response distinguishes light touch from no SDK depth."""
    mask = depth > 0
    magnitudes = np.linalg.norm(shear[mask], axis=-1) if mask.any() else np.array([0.0])
    return dict(
        depth_peak_mm=float(np.max(depth)),
        response_pixels=int(mask.sum()),
        shear_peak_nominal_mm=float(np.max(magnitudes) * DMV1_CODE_SHEAR_TO_M * 1000),
    )


def draw_history(panel, history, zero, limit=15.0):
    import cv2

    left, top, width, height = 24, 418, 910, 72
    cv2.rectangle(panel, (left, top), (left + width, top + height), (48, 43, 38), -1)
    if not history:
        return
    now = history[-1][0]
    recent = [(t, v) for t, v in history if t >= now - 10]
    values = np.array([np.degrees(v - zero) for _, v in recent])
    colors = [(110, 160, 255), (140, 225, 110), (255, 195, 95)]
    cv2.line(panel, (left, top + height // 2), (left + width, top + height // 2), (90, 80, 70), 1)
    for axis, color in enumerate(colors):
        previous = None
        for (stamp, _), vector in zip(recent, values):
            if not np.isfinite(vector[axis]):
                previous = None
                continue
            point = (
                left + round(width * (stamp - now + 10) / 10),
                top + round(height * (0.5 - 0.45 * np.clip(vector[axis] / limit, -1, 1))),
            )
            if previous is not None:
                cv2.line(panel, previous, point, color, 2, cv2.LINE_AA)
            previous = point
        cv2.putText(
            panel,
            ["Rx", "Ry", "Rz"][axis],
            (left + axis * 60, top - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    clipped = np.count_nonzero(np.isfinite(values) & (np.abs(values) > limit))
    note = " | off-scale values" if clipped else " | gaps = unavailable"
    cv2.putText(
        panel,
        f"10 s | fixed +/- {limit:g} deg" + note,
        (270, top - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (185, 190, 200),
        1,
        cv2.LINE_AA,
    )


def render_panel(
    image, depth, shear, result, serial, zero=None, history=None, chart_limit_deg=15.0
):
    import cv2

    image = np.asarray(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        image = image.astype(np.uint8).copy()
    h, w = depth.shape
    heat = cv2.applyColorMap(np.uint8(np.clip(depth / 3.5, 0, 1) * 255), cv2.COLORMAP_TURBO)
    arrows = image.copy()
    # Physical conversion followed by pixel conversion, with explicit display gain.
    vectors = shear * DMV1_CODE_SHEAR_TO_M * result["assumptions"]["ppmm"] * 1000 * 5
    for y in range(10, h, 18):
        for x in range(10, w, 18):
            if depth[y, x] > result["assumptions"]["threshold_mm"]:
                d = np.clip(vectors[y, x], -40, 40)
                cv2.arrowedLine(
                    arrows,
                    (x, y),
                    (round(x + d[0]), round(y + d[1])),
                    (70, 240, 160),
                    1,
                    tipLength=0.3,
                )
    tiles = [cv2.resize(tile, (320, 240)) for tile in (image, heat, arrows)]
    panel = np.full((505, 960, 3), (28, 22, 18), np.uint8)
    display = result.get("display")
    state = (
        display["state"] if display else ("tracking" if "rotation_vector_rad" in result else "idle")
    )
    accent = (
        (155, 220, 100)
        if state == "tracking"
        else (100, 195, 255) if state != "idle" else (145, 130, 115)
    )
    cv2.rectangle(panel, (0, 0), (5, 505), accent, -1)
    panel[70:310] = np.hstack(tiles)

    def label(text, xy, size=0.6, color=(225, 230, 240)):
        cv2.putText(panel, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)

    label(serial, (20, 28), 0.75)
    metrics = response_metrics(depth, shear)
    label("TACTILE DEMO / local response", (470, 28), 0.5)
    for x, title in zip(
        (20, 340, 660), ("TACTILE IMAGE", "DEPTH / nominal mm", "SHEAR / arrows x5")
    ):
        label(title, (x, 57), 0.5)
    vector = display["rotation_vector_rad"] if display else result.get("rotation_vector_rad")
    if vector is None:
        status = (
            "LIGHT RESPONSE / below estimate threshold"
            if metrics["response_pixels"]
            else "WAITING FOR CONTACT"
        )
        status = {
            "settling": "CONTACT / stabilizing",
            "small_contact": "SMALL CONTACT",
            "weak_fit": "ESTIMATE UNRELIABLE",
            "invalid_estimate": "ESTIMATE UNAVAILABLE",
        }.get(state, status)
        label(status, (20, 340), color=(100, 195, 255))
        label("Local response unavailable", (20, 367), 0.5)
    else:
        degrees = np.degrees(np.array(vector) - (np.zeros(3) if zero is None else zero))
        mode = result.get("display_mode", "raw").upper()
        label(mode + "  Rx %+.2f   Ry %+.2f   Rz %+.2f deg" % tuple(degrees), (20, 340))
        label("LOCAL RESPONSE / nominal calibration", (20, 367), 0.5)
        if result["flags"]:
            label("Estimate outside nominal fit range", (510, 367), 0.5, (100, 195, 255))
    draw_history(panel, history, np.zeros(3) if zero is None else zero, chart_limit_deg)
    return panel
