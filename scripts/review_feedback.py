"""Compare raw/display step changes on the same recorded frames, without hardware."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from tacse3.demo_feedback import DisplayFeedback


def review(rows):
    report = {}
    serials = sorted({row["serial"] for row in rows if "serial" in row})
    for serial in serials:
        samples = [row for row in rows if row.get("serial") == serial]
        feedback = DisplayFeedback()
        counts = Counter()
        previous = None
        raw_steps, smooth_steps = [], []
        for row in samples:
            display = feedback.update(row)
            counts[display["state"]] += 1
            value = display["rotation_vector_rad"]
            if value is None:
                previous = None
                continue
            raw = np.array(row["rotation_vector_rad"])
            value = np.array(value)
            if previous is not None and 0 < row["time"] - previous[0] <= feedback.config.max_gap_s:
                raw_steps.append(float(np.degrees(np.linalg.norm(raw - previous[1]))))
                smooth_steps.append(float(np.degrees(np.linalg.norm(value - previous[2]))))
            previous = row["time"], raw, value

        def stats(values):
            return (
                dict(p95_deg=float(np.percentile(values, 95)), maximum_deg=max(values))
                if values
                else None
            )

        report[serial] = dict(
            samples=len(samples),
            display_states=dict(counts),
            matched_step_pairs=len(raw_steps),
            raw_steps=stats(raw_steps),
            smooth_steps=stats(smooth_steps),
        )
    return dict(
        description="Display smoothness only, matched consecutive frames; not physical accuracy.",
        filter_config=vars(DisplayFeedback().config),
        sensors=report,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    skipped = 0
    for line in args.log.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            skipped += 1
    result = review(rows)
    result["malformed_lines_skipped"] = skipped
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
