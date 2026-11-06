<p align="center"><img src="docs/assets/banner.svg" alt="TacSE3 — tactile motion, made visible" width="920"></p>

<p align="center"><strong>Live tactile feedback · Recorded replay · SE(3) estimation</strong><br>
Python 3.10+ · NumPy / SciPy · Optional Daimon SDK and LeRobot integration</p>

<p align="center"><a href="README.zh-CN.md">中文</a> · <a href="docs/ball-demo.md">Demo guide</a> · <a href="docs/architecture.md">Architecture</a> · <a href="CONTRIBUTING.md">Contributing</a></p>

TacSE3 reads tactile depth and shear, estimates motion, and makes the response
visible. Start with a hand-turned ball between stationary tactile fingertips:
watch contact maps, shear arrows and per-sensor rotation traces without connecting
a robot driver. The original monitoring and optional UR5 paths remain available.

<p align="center"><img src="docs/assets/ball-demo.png" alt="Two tactile sensors responding during a hand-turned ball demonstration" width="660"></p>

*Tactile images, depth maps, shear arrows and per-sensor rotation histories.*

## Choose a workflow

| Workflow | Entry | Needs hardware? |
| --- | --- | --- |
| Interactive ball demo | `python -m tacse3.ball_demo` | Tactile sensors only |
| Bounded observation / recording | `python -m tacse3.observe` | Tactile sensors only |
| Saved-frame numerical replay | `python -m tacse3.offline_replay` | No |
| Original monitor / optional robot integration | `python -m tacse3.run_monitor` | Depends on mode |

## Quick start

Install into your existing sensor environment for live use. Install the vendor's
`dmrobotics` SDK separately; it is not bundled in this repository.

```bash
pip install -e ".[demo]"
python -m tacse3.ball_demo --sensor-serial SERIAL1 SERIAL2
```

Use a desktop terminal with GUI-enabled OpenCV.

1. Wait for sensor initialization with the surfaces unloaded.
2. Place a suitable ball between the stationary fingertips and turn it by hand.
3. Press **Z** to zero the display, **R** to compare raw/smooth feedback,
   **S** to save frames, or **Q** to quit.

Snapshots of the strongest response and the latest display are saved automatically.
Raw estimates stay in the log alongside display-filtered values. The display
filter is independent of estimation, fusion and robot-control code.

For a saved NPZ, no sensor SDK or desktop is needed:

```bash
pip install -e ".[headless]"  # use in an offline environment, instead of [demo]
python -m tacse3.ball_demo --sample sample.npz --preview-output preview.png
python -m tacse3.offline_replay sample.npz \
  --depth-unit mm --shear-unit dmv1-code --mm-per-pixel 0.0634 --output replay.json
```

Choose one OpenCV variant per environment. `[viz]` adds GUI OpenCV and Open3D
for the original visualizer. `[headless]` is for image export and CI.

## Project structure

```text
tacse3/
  ball_demo.py                 Interactive sensor-only demo
  demo_view.py                 Image panels and history charts
  demo_feedback.py             Display state and smoothing
  observe.py                   Sensor recording and reports
  offline_replay.py            Saved-array replay
  run_monitor.py              Main monitor CLI
  sensor_loop.py               Monitor orchestration
  contact_center_translation.py  Contact-centroid translation
  pose_visualizer.py           Open3D visualization
  core/                       Estimation, units, fusion and robot integration
docs/                         Guides and architecture
scripts/                      Launch and analysis utilities
tests/                        Hardware-free tests
.github/workflows/            Continuous integration
```

The existing shear estimator, contact-centroid translation, dual-sensor fusion,
legacy monitor, Open3D visualization and UR5 worker remain in the repository.
Display stabilization only affects the ball-demo view. Read the
[monitor and recording guide](docs/legacy-usage.md) for those entry points.

Nominal SDK units are documented in [unit conventions](UNIT_CALIBRATION.md).
The ball demo is qualitative feedback, not a calibrated measurement of a ball's
absolute orientation.

## Develop and verify

```bash
pip install -e ".[headless,dev]"
python -B -m unittest discover -s tests -v
python -m build
```

Tests use synthetic arrays, recorded-data-compatible functions and fake devices.
The CI workflow runs offline checks on Python 3.10 and 3.12; hardware is not needed.
See [contributing](CONTRIBUTING.md), [architecture](docs/architecture.md),
[display behavior](docs/ball-demo.md) and [changes](CHANGELOG.md).

## Origins and license

Based on [yuanzero/tacSE3_code](https://github.com/yuanzero/tacSE3_code), factored
from `realtime_daimon_rot`. The package retains its existing Apache-2.0 license
declaration; the [license text](LICENSE) is included. Vendor SDKs and external
robot drivers remain separately installed components with their own terms.
