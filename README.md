# TacSE3 · Tactile Demo

Live tactile images, contact depth, shear response and per-sensor 3D feedback.
A prototype for exploring contact and local motion responses with a hand-turned ball.

[中文](README.zh-CN.md) · [Demo guide](docs/ball-demo.md)

## Start

Use a desktop terminal with the vendor `dmrobotics` SDK installed separately.

```bash
pip install -e ".[viz]"
python -m tacse3.ball_demo --sensor-serial SERIAL1 SERIAL2 --pose3d on
```

Replace the serials with your devices. Start with unloaded sensors, wait for
initialization, then place the ball between the stationary sensor surfaces and
turn it gently. For 2D only, install `.[demo]` and use `--pose3d off`.

| Key | Action |
| --- | --- |
| Z | Zero the display |
| R | Switch raw / smooth |
| S | Save a snapshot |
| Q / Esc | Exit |

Focus the 2D dashboard for keyboard controls. Grey 3D axes are the reference;
colored axes show the current local response. Recordings are saved under `runs/`.

The displayed values use nominal sensor calibration and describe local responses,
not an absolute or accumulated object pose. See [units and calibration](UNIT_CALIBRATION.md).

## More

- [Display options and recording details](docs/ball-demo.md)
- [Existing monitor and robot integration](docs/legacy-usage.md)
- [Architecture](docs/architecture.md) · [Development](CONTRIBUTING.md)

Based on [yuanzero/tacSE3_code](https://github.com/yuanzero/tacSE3_code).
[Apache-2.0](LICENSE). Vendor SDKs and robot drivers are separate dependencies.
