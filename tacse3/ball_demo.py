"""Hand-turned ball demonstration: tactile images and per-sensor estimates only."""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
from .offline_replay import replay
from .core.units import NOMINAL_MM_PER_PIXEL, pixels_per_mm
from .demo_view import render_panel, response_metrics
from .demo_feedback import DisplayFeedback, FeedbackConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-serial", nargs="+", help="Connected tactile sensor serials")
    parser.add_argument("--mm-per-pixel", type=float, default=NOMINAL_MM_PER_PIXEL)
    parser.add_argument("--threshold-mm", type=float, default=0.5)
    parser.add_argument("--sample", type=Path, help="Offline NPZ preview; does not load SDK")
    parser.add_argument(
        "--preview-output", type=Path, help="Write one offline preview PNG without opening a window"
    )
    parser.add_argument("--output", type=Path, default=Path("runs") / f"ball-{time.time_ns()}")
    parser.add_argument(
        "--smoothing-ms",
        type=float,
        default=120.0,
        help="Display EMA time constant; 0 disables median/EMA",
    )
    parser.add_argument(
        "--min-contact-pixels", type=int, default=100, help="Display-only minimum contact area"
    )
    parser.add_argument(
        "--activation-ms", type=float, default=60.0, help="Display contact confirmation time"
    )
    parser.add_argument(
        "--chart-limit-deg",
        type=float,
        default=15.0,
        help="Fixed chart half range; does not clamp estimates",
    )
    parser.add_argument(
        "--raw-display", action="store_true", help="Start with unfiltered estimator display"
    )
    args = parser.parse_args(argv)
    if args.preview_output and not args.sample:
        parser.error("--preview-output requires --sample")
    if not args.sample and not args.sensor_serial:
        parser.error("--sensor-serial is required for live acquisition")
    if args.sensor_serial:
        if len(set(args.sensor_serial)) != len(args.sensor_serial):
            parser.error("sensor serials must be distinct")
        if any(
            not serial or not all(c.isascii() and (c.isalnum() or c in "-_") for c in serial)
            for serial in args.sensor_serial
        ):
            parser.error("sensor serials may contain only ASCII letters, digits, '-' and '_'")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be a new or empty directory")
    if not np.isfinite(args.threshold_mm) or args.threshold_mm < 0:
        parser.error("--threshold-mm must be finite and nonnegative")
    if not np.isfinite(args.chart_limit_deg) or args.chart_limit_deg <= 0:
        parser.error("--chart-limit-deg must be finite and positive")
    try:
        feedback_config = FeedbackConfig(
            args.smoothing_ms, args.min_contact_pixels, args.activation_ms
        )
        ppmm = pixels_per_mm(args.mm_per_pixel)
    except ValueError as error:
        parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    import cv2

    sensors, zeros, histories, peaks, feedback = [], {}, {}, {}, {}
    # A still image has no temporal history to smooth.
    raw_mode = args.raw_display or args.sample is not None
    last_preview = 0.0
    last_tick = time.monotonic()
    fps = 0.0
    try:
        if args.sample:
            with np.load(args.sample, allow_pickle=False) as data:
                saved = tuple(data[k].copy() for k in ("image", "depth", "shear"))
            serials = ["OFFLINE SAMPLE"]
        else:
            from dmrobotics import Sensor

            serials = args.sensor_serial
            for serial in serials:
                sensor = Sensor(serial)
                sensors.append(sensor)
                start = time.monotonic()
                while sensor.getStatus() != 0:
                    if time.monotonic() - start > 45:
                        raise RuntimeError(f"{serial}: initialization timed out")
                    time.sleep(0.1)
        (args.output / "session.json").write_text(
            json.dumps(
                dict(
                    schema_version=2,
                    serials=serials,
                    mm_per_pixel=args.mm_per_pixel,
                    threshold_mm=args.threshold_mm,
                    feedback=vars(feedback_config),
                    chart_limit_deg=args.chart_limit_deg,
                    sample=str(args.sample) if args.sample else None,
                    actuator_commands_sent=0,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        if not args.preview_output:
            cv2.namedWindow("TacSE3 | Hand-turned ball", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
            cv2.resizeWindow("TacSE3 | Hand-turned ball", 960, min(1000, len(serials) * 505 + 65))
        print(
            (
                "Offline preview"
                if args.sample
                else "Ready: Z zero | R raw/smooth | S snapshot | Q quit"
            ),
            flush=True,
        )
        with (args.output / "estimates.jsonl").open("a", encoding="utf-8") as log:
            while True:
                panels, results, frames = [], {}, {}
                for i, serial in enumerate(serials):
                    if args.sample:
                        image, depth, shear = saved
                    else:
                        sensor = sensors[i]
                        image, depth, shear = (
                            np.asarray(x).copy()
                            for x in (sensor.getRawImage(), sensor.getDepth(), sensor.getShear())
                        )
                    result = replay(
                        depth,
                        shear,
                        depth_unit="mm",
                        shear_unit="dmv1-code",
                        ppmm=ppmm,
                        threshold_mm=args.threshold_mm,
                    )
                    result.update(serial=serial, time=time.time())
                    result.update(response_metrics(depth, shear))
                    stabilized = feedback.setdefault(
                        serial, DisplayFeedback(feedback_config)
                    ).update(result)
                    # Raw and filtered data coexist in the log, independent of the selected view.
                    result["filtered_display"] = stabilized
                    result["display_mode"] = "raw" if raw_mode else "smooth"
                    result["display"] = (
                        dict(
                            state=(
                                "tracking"
                                if "rotation_vector_rad" in result
                                else ("light_response" if result["response_pixels"] else "idle")
                            ),
                            rotation_vector_rad=result.get("rotation_vector_rad"),
                        )
                        if raw_mode
                        else stabilized
                    )
                    log.write(json.dumps(result) + "\n")
                    results[serial] = result
                    frames[serial] = dict(image=image, depth=depth, shear=shear)
                    history = histories.setdefault(serial, deque(maxlen=500))
                    vector = result["display"]["rotation_vector_rad"]
                    history.append(
                        (result["time"], np.array(vector if vector is not None else [np.nan] * 3))
                    )
                    panels.append(
                        render_panel(
                            image,
                            depth,
                            shear,
                            result,
                            serial,
                            zeros.get(serial),
                            history,
                            args.chart_limit_deg,
                        )
                    )
                    # Bounded storage: one strongest response per sensor, overwritten in place.
                    score = result["shear_peak_nominal_mm"]
                    if score > peaks.get(serial, 0.0):
                        peaks[serial] = score
                        np.savez_compressed(args.output / f"peak-{serial}.npz", **frames[serial])
                        (args.output / f"peak-{serial}.json").write_text(
                            json.dumps(result, indent=2), encoding="utf-8"
                        )
                        cv2.imwrite(str(args.output / f"peak-{serial}.png"), panels[-1])
                tick = time.monotonic()
                fps = (
                    0.9 * fps + 0.1 / max(tick - last_tick, 1e-6)
                    if fps
                    else 1 / max(tick - last_tick, 1e-6)
                )
                last_tick = tick
                footer = np.full((65, 960, 3), (28, 22, 18), np.uint8)
                cv2.putText(
                    footer,
                    "TACSE3 | Z zero  R raw/smooth  S save  Q quit | %.1f Hz | auto-save" % fps,
                    (20, 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (160, 225, 180),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    footer,
                    "Nominal per-sensor estimates; not an absolute ball orientation.",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (170, 175, 185),
                    1,
                    cv2.LINE_AA,
                )
                canvas = np.vstack(panels + [footer])
                if tick - last_preview > 2:
                    cv2.imwrite(str(args.output / "latest-display.jpg"), canvas)
                    last_preview = tick
                if args.preview_output:
                    if not cv2.imwrite(str(args.preview_output), canvas):
                        raise RuntimeError("Could not write preview image")
                    break
                cv2.imshow("TacSE3 | Hand-turned ball", canvas)
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("z"):
                    for serial, result in results.items():
                        if result["display"]["rotation_vector_rad"] is not None:
                            zeros[serial] = np.array(result["display"]["rotation_vector_rad"])
                    log.write(
                        json.dumps(
                            dict(
                                event="display_zero",
                                time=time.time(),
                                references={k: v.tolist() for k, v in zeros.items()},
                            )
                        )
                        + "\n"
                    )
                if key == ord("r"):
                    raw_mode = not raw_mode
                    histories.clear()
                    zeros.clear()
                    log.write(
                        json.dumps(
                            dict(
                                event="display_mode",
                                time=time.time(),
                                mode="raw" if raw_mode else "smooth",
                                zero_cleared=True,
                            )
                        )
                        + "\n"
                    )
                if key == ord("s"):
                    stamp = time.time_ns()
                    for serial, data in frames.items():
                        np.savez_compressed(args.output / f"{stamp}-{serial}.npz", **data)
                    cv2.imwrite(str(args.output / f"{stamp}-display.png"), canvas)
                log.flush()
    finally:
        for sensor in sensors:
            sensor.disconnect()
        if not args.preview_output:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
