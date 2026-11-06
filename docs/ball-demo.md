# Hand-turned ball demo

[Home](../README.md) · [中文](../README.zh-CN.md) · [Unit conventions](../UNIT_CALIBRATION.md)

## Operation

Launch from a desktop terminal after installing GUI-enabled OpenCV and the
sensor SDK. Start unloaded, let SDK initialization complete, then place the ball
between the stationary sensor surfaces. Turn gently in each direction, pause,
and reverse. This workflow does not instantiate robot or gripper drivers.

```bash
python -m tacse3.ball_demo --sensor-serial SERIAL1 SERIAL2
```

| Key | Action |
| --- | --- |
| Z | Zero the current displayed rotation vectors; does not reset the SDK |
| R | Switch raw/smooth display; clear display zero and chart to avoid mixing modes |
| S | Save sensor arrays and full window image |
| Q / Esc | Close the demo and release the sensors |

Each sensor has its own contact image, depth map, shear-arrow overlay and
10-second history. The shear arrows have a display gain of 5, and long arrows
are clipped for readability. They are not force arrows. Depth uses a fixed
0–3.5 nominal mm color range. Rotation traces use a fixed range, preventing a
single large estimate from repeatedly resizing the plot. Off-scale traces are
marked; numerical values are not clamped.

## Display filtering

The estimator still receives the same depth and shear with the same 0.5 nominal
mm contact threshold. Stabilization is downstream, in `demo_feedback.py`:

1. Check availability, fit rank and display contact area.
2. Confirm a new contact briefly, then display its response.
3. Apply a three-frame median and time-aware exponential moving average.
4. Clear the filter immediately on missing/small contact or a long time gap.

This adds roughly a few frames of lag. It improves readability, not measured
pose accuracy. Raw mode bypasses the display gates and filter and keeps the
original estimates visible for comparison. A static NPZ preview always shows
raw estimates because it has no temporal history.

| Option | Default | Meaning |
| --- | --- | --- |
| `--threshold-mm` | 0.5 | Existing estimator contact threshold |
| `--smoothing-ms` | 120 | EMA time constant; 0 disables median and EMA |
| `--activation-ms` | 60 | New-contact confirmation time |
| `--min-contact-pixels` | 100 | Display-only contact-area minimum |
| `--chart-limit-deg` | 15 | Plot half range; does not alter estimates |
| `--raw-display` | off | Start in raw comparison mode |
| `--mm-per-pixel` | 0.0634 | Nominal historical spatial pitch |
| `--output` | unique directory under runs | New or empty session directory |

Light response below threshold, small contact, settling, tracking and weak fit
are distinguished. Missing estimates leave chart gaps rather than fabricated
zero angles. Values are per-sensor estimates, not a fused absolute ball pose.

## Saved files

| File | Contents |
| --- | --- |
| `session.json` | Schema version, serials, nominal scale and display settings |
| `estimates.jsonl` | Raw estimates, response metrics, filtered display and keyboard events |
| `latest-display.jpg` | Most recent full display, updated every two seconds |
| `peak-SERIAL.npz` | Image, depth and shear at strongest observed shear response |
| `peak-SERIAL.json` / `.png` | Matching result and panel at that peak |
| `TIMESTAMP-SERIAL.npz` / `TIMESTAMP-display.png` | Manual snapshots |

One peak per sensor is overwritten when a stronger response arrives. Peaks for
different sensors can occur at different times. They must not be treated as a
synchronized pair. SDK getters are sequential, not hardware-synchronized.
Logs retain the raw `rotation_vector_rad`; `filtered_display` is the independent
filtered estimate, and `display` records the selected view. Z stores a display
reference event; raw and filtered vectors remain unzeroed in the log.

## Reproduce a display comparison

```bash
PYTHONPATH=. python scripts/review_feedback.py runs/SESSION/data/estimates.jsonl \
  --output feedback-review.json
```

The review compares raw and smooth frame-to-frame changes on the same eligible
consecutive frame pairs. It reports omitted display states separately. A lower
step-change percentile is a smoothness metric, not a physical accuracy score.

## Troubleshooting

- No window: use a desktop session and GUI OpenCV (`.[demo]`), or use
  `--sample ... --preview-output ...` without a display.
- No contact: watch depth/response pixels first. Do not lower the threshold just
  to force a rotation estimate. `LIGHT RESPONSE` is still valid tactile feedback.
- Device busy: close other readers before starting another live session.
- Want original behavior: press R. Existing monitor paths are unchanged by the
  display filter and remain documented in [legacy usage](legacy-usage.md).
