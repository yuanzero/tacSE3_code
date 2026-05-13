# TacSE3

**TacSE3** is a small, publishable bundle for realtime **tactile** sensing and **SE(3)**-style motion estimation (shear + depth, optional in-plane translation from contact centroids), with optional **UR5** command via LeRobot. It was factored out of `realtime_daimon_rot/run_realtime_monitor.py` and `apply_ur5_rotation.py`.

- **Repository / project folder:** `TacSE3/`
- **Python import package:** `tacse3` (lowercase, valid module name)

## What it does

- Reads Daimon-style sensors via `dmrobotics`.
- Estimates rigid-body motion from shear + depth (optional contact-centroid XY translation).
- Optionally commands a Universal Robots arm through **LeRobot** (`UR5Robot` / `UR5Config`) when `--execute` is set.
- Optional OpenCV stream + Open3D `PoseVisualizer` when `--visualize` is set.

## Install

From the `TacSE3` directory (the folder that contains this `README.md` and `pyproject.toml`):

```bash
cd TacSE3
pip install -e .
# optional UI stacks
pip install -e ".[viz]"
```

`dmrobotics` and LeRobot are environment-specific; install them the same way you do for the parent project.

## Run

```bash
python -m tacse3.run_monitor --sensor-serial "SERIAL1 SERIAL2" --visualize --duration 60

python -m tacse3 --sensor-serial "SERIAL1 SERIAL2"   # same via __main__
```

For example: python -m tacse3.run_monitor --sensor-serial "M2505150016 M2505150299" --visualize --duration 60


After `pip install`, you can also use the console script:

```bash
tacse3-monitor --sensor-serial "SERIAL1 SERIAL2" --help
```

Headless fast path (no centroid translation, no OpenCV) matches the refactored `DaimonMonitor` branch inside `sensor_loop.simple_sensor_loop_cli`. Use `--legacy-daimon-monitor` for the older monitor-only loop.

## Layout

| Path | Role |
|------|------|
| `tacse3/run_monitor.py` | CLI mirroring the original `run_realtime_monitor.py` |
| `tacse3/sensor_loop.py` | Defaults, argparse helpers, `simple_sensor_loop_cli` |
| `tacse3/contact_center_translation.py` | Mask-centroid translation state |
| `tacse3/pose_visualizer.py` | Open3D pose / cumulative-angle view |
| `tacse3/core/` | `DaimonMonitor`, shear estimator, fusion, UR5 action math, background worker |

## License

Apache-2.0 (match the parent repository if you fork).
