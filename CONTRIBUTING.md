# Contributing

TacSE3 welcomes focused improvements to tactile estimation, visualization and
reproducibility. Keep existing monitor, centroid-translation, fusion and robot
integration paths available when improving the sensor-only workflow.

## Development

Use Python 3.10 or newer. For desktop visualization install `pip install -e
".[demo,dev]"`; for offline/CI work use `pip install -e ".[headless,dev]"`.
Choose one OpenCV variant. The vendor SDK and LeRobot are external integrations,
not dependencies of the numerical tests.

```bash
python -B -m unittest discover -s tests -v
python -m black tacse3/ball_demo.py tacse3/demo_view.py tacse3/demo_feedback.py \
  scripts/review_feedback.py tests/test_ball_demo.py tests/test_demo_feedback.py
python -m build
```

Formatting is initially scoped to the demo code so unrelated legacy code does
not acquire a large formatting-only diff.

## Pull requests

Describe the observed problem, resulting behavior and relevant checks. Include
before/after images for interface changes. State whether results are synthetic,
recorded replay or live hardware observations. Keep raw estimates available
when adding display filtering; do not describe smoother plots as improved
physical accuracy.

Do not include local session logs, binary caches, machine credentials or the
vendor SDK in a contribution. Small representative screenshots belong in
`docs/assets/`. The original robot integration is retained, but the automated
test workflow must run without physical devices.
