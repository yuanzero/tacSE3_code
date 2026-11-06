# Monitor and recording interfaces

[Project overview](../README.md) · [Ball demo](ball-demo.md) · [Architecture](architecture.md)

## Monitor

The main monitor combines sensor preprocessing, shear-based rotation estimation,
optional contact-centroid translation, fusion and visualization.

```bash
python -m tacse3.run_monitor --sensor-serial "SERIAL1 SERIAL2" --visualize --duration 60
python -m tacse3 --sensor-serial "SERIAL1 SERIAL2" --duration 60
```

The installed `tacse3-monitor` command exposes the same CLI. Pass `--help` for
all available options. This CLI accepts sensor serials as one quoted string;
the ball-demo and observer CLIs accept separate arguments.

| Option | Purpose |
| --- | --- |
| `--visualize` | Open the monitor's visual feedback |
| `--duration`, `--interval` | Configure run length and loop interval |
| `--smooth-window` | Configure the monitor's existing window-based smoothing |
| `--no-contact-center-translation` | Use shear-estimator translation instead of centroid translation |
| `--legacy-daimon-monitor` | Select the headless legacy monitor implementation |
| `--ppmm`, `--depth-in-mm`, `--depth-in-m` | Specify input scale and depth convention |
| `--axis-order`, `--axis-sign` | Configure axis mapping |

The ball demo's display-only filter is independent of monitor smoothing.
The optional UR5 integration uses external LeRobot drivers and is enabled with
`--execute`. Use the sensor-only demo or observer when no actuator connection is
wanted. This project does not bundle robot-driver initialization or calibration.

## Sensor-only recording

```bash
python -m tacse3.observe --sensor-serial SERIAL1 SERIAL2 \
  --duration 60 --output runs/observation-01
```

The observer saves JSONL metrics, per-sensor peak arrays, SDK logs and an HTML
summary in a new or empty output directory. It exposes startup and stream-stall
timeouts and can be stopped with Ctrl+C. The sensor SDK is installed separately.
Run from a writable directory because the SDK can create its own log file.

## Offline replay

```bash
python -m tacse3.offline_replay sample.npz \
  --depth-unit mm --shear-unit dmv1-code --mm-per-pixel 0.0634 --output replay.json
```

The NPZ contains `depth` and `shear`; image previews additionally use `image`.
Replay does not load a sensor SDK or robot driver. See [unit conventions](../UNIT_CALIBRATION.md)
for supported scales and the [demo guide](ball-demo.md) for saved-file formats.
