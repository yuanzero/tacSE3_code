"""Bounded, sensor-only acquisition. No robot or gripper objects are created."""
from __future__ import annotations

import argparse
import contextlib
import html
import json
import multiprocessing as mp
import queue
import time
from pathlib import Path

import numpy as np


def write_report(directory, report):
    cards = []
    for serial, values in report["sensors"].items():
        cards.append(f'''<article><h2>{html.escape(serial)}</h2>
          <div class="number">{values['samples']:,}<small>有效采样组</small></div>
          <dl><dt>非零深度采样</dt><dd>{values['contact_samples']:,}</dd>
          <dt>深度峰值 · SDK 单位</dt><dd>{values['depth_peak_sdk']:.5f}</dd></dl>
          <a href="sdk-{serial}.log">SDK 日志 ↗</a> · <a href="peak-{serial}.npz">峰值帧数据 ↗</a></article>''')
    error = html.escape(report["error"] or "无采集错误")
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>TacSE3 · 采集报告</title>
    <style>body{margin:0;background:#0d1421;color:#e7edf6;font:16px/1.6 system-ui,sans-serif}
    main{max-width:1000px;margin:auto;padding:48px 24px}header{border-bottom:1px solid #334155;padding-bottom:28px}
    .tag{color:#74e1bd;letter-spacing:.15em;font-size:12px}h1{font-size:40px;margin:8px 0}
    h2{font-size:19px;margin:0;color:#bcd0eb}.muted,small,dt{color:#9aacc6}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:20px;margin:28px 0}
    article{background:#172235;border:1px solid #2a3a52;border-radius:16px;padding:28px}
    .number{font-size:48px;font-weight:600;margin:18px 0}small{display:block;font-size:14px;font-weight:400}
    dl{display:grid;grid-template-columns:1fr auto;gap:10px}dd{margin:0}a{color:#8ac6ff;text-decoration:none}
    aside{border-left:3px solid #74e1bd;padding:4px 20px;color:#bcd0eb}footer{margin-top:32px;color:#9aacc6}
    </style><main><header><div class="tag">TACSE3 / SENSOR OBSERVATION</div>
    <h1>双传感器采集报告</h1><p class="muted">仅观测模式 · 无机械臂或夹爪指令</p></header>'''
    document += f'<p>状态：{html.escape(report["status"])} · 总耗时 {report["elapsed_seconds"]:.1f} 秒 · {error}</p>'
    document += '<section class="grid">' + ''.join(cards) + '</section>'
    document += '<aside>接触状态仅依据 SDK 深度是否非零。采集通过不等于深度标定、SE(3) 位姿估计或闭环控制已验证。</aside>'
    document += '<footer><a href="samples.jsonl">完整采样记录 ↗</a> · <a href="summary.json">JSON 摘要 ↗</a></footer></main></html>'
    (Path(directory) / "report.html").write_text(document, encoding="utf-8")


def frame_metrics(image, depth, shear):
    arrays = [np.asarray(x) for x in (image, depth, shear)]
    image, depth, shear = arrays
    if depth.ndim != 2 or image.shape[:2] != depth.shape or shear.shape != (*depth.shape, 2):
        raise ValueError("image/depth/shear shapes do not match")
    if any(a.size == 0 or not np.isfinite(a).all() for a in arrays):
        raise ValueError("empty or nonfinite sensor data")
    return {
        "depth_peak_sdk": float(np.max(np.abs(depth))),
        "depth_nonzero_pixels": int(np.count_nonzero(depth)),
        "shear_abs_mean_sdk": float(np.abs(shear).mean()),
        "shape": list(depth.shape),
    }


def _worker(serial, output, stop, directory, interval):
    """A separate process allows the parent to bound a blocked vendor call."""
    sensor = None
    directory = Path(directory)
    try:
        with (directory / f"sdk-{serial}.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            from dmrobotics import Sensor

            sensor = Sensor(serial)
            while not stop.is_set() and sensor.getStatus() != 0:
                stop.wait(.5)
            if stop.is_set():
                return
            output.put({"serial": serial, "event": "connected", "time": time.time()})
            peak = -1.0
            while not stop.is_set():
                start = time.monotonic()
                image = np.asarray(sensor.getRawImage()).copy()
                depth = np.asarray(sensor.getDepth()).copy()
                shear = np.asarray(sensor.getShear()).copy()
                record = frame_metrics(image, depth, shear)
                record.update(serial=serial, event="sample", time=time.time())
                if record["depth_peak_sdk"] > peak:
                    peak = record["depth_peak_sdk"]
                    np.savez_compressed(directory / f"peak-{serial}.npz", image=image, depth=depth, shear=shear)
                output.put(record, timeout=2)
                stop.wait(max(0, interval - (time.monotonic() - start)))
    except Exception as error:
        output.put({"serial": serial, "event": "error", "message": str(error), "time": time.time()}, timeout=2)
    finally:
        if sensor is not None:
            sensor.disconnect()


def main(argv=None):
    parser = argparse.ArgumentParser(description="TacSE3 sensor-only observer; SDK units, no actuator control.")
    parser.add_argument("--sensor-serial", nargs="+", required=True)
    parser.add_argument("--duration", type=float, default=60, help="Recording seconds after both sensors deliver data")
    parser.add_argument("--interval", type=float, default=.05)
    parser.add_argument("--startup-timeout", type=float, default=45)
    parser.add_argument("--stall-timeout", type=float, default=8)
    parser.add_argument("--output", type=Path, required=True, help="New, empty session directory")
    args = parser.parse_args(argv)
    serials = [s for token in args.sensor_serial for s in token.replace(",", " ").split()]
    if not serials or len(set(serials)) != len(serials) or any(not s.isalnum() for s in serials):
        parser.error("provide distinct alphanumeric serial numbers")
    if any(not np.isfinite(v) or v <= 0 for v in [args.duration, args.interval, args.startup_timeout, args.stall_timeout]):
        parser.error("durations and intervals must be positive and finite")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    messages = ctx.Queue(maxsize=256)
    stop = ctx.Event()
    processes = []
    stats = {s: {"samples": 0, "contact_samples": 0, "depth_peak_sdk": 0.0} for s in serials}
    latest, seen = {}, {}
    began = time.monotonic()
    recording_start = None
    printed = 0
    failure = None
    print("TacSE3 | SENSOR ONLY | no actuator commands", flush=True)
    print("Contact = nonzero SDK depth; physical units and pose calibration are not validated.", flush=True)
    try:
        for serial in serials:
            process = ctx.Process(target=_worker, args=(serial, messages, stop, str(args.output.resolve()), args.interval))
            process.start()
            processes.append(process)
        with (args.output / "samples.jsonl").open("w") as stream:
            while True:
                now = time.monotonic()
                if recording_start is not None and now - recording_start >= args.duration:
                    break
                if recording_start is None and now - began > args.startup_timeout:
                    raise RuntimeError("sensor startup timed out")
                if any(now - t > args.stall_timeout for t in seen.values()):
                    raise RuntimeError("sensor stream stalled")
                if any(p.exitcode is not None for p in processes):
                    raise RuntimeError("sensor worker exited unexpectedly; inspect SDK logs")
                try:
                    record = messages.get(timeout=.2)
                except queue.Empty:
                    continue
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                serial = record["serial"]
                if record["event"] == "error":
                    raise RuntimeError(f"{serial}: {record['message']}")
                if record["event"] != "sample":
                    continue
                seen[serial] = time.monotonic()
                latest[serial] = record
                stats[serial]["samples"] += 1
                stats[serial]["contact_samples"] += int(record["depth_nonzero_pixels"] > 0)
                stats[serial]["depth_peak_sdk"] = max(stats[serial]["depth_peak_sdk"], record["depth_peak_sdk"])
                if len(seen) == len(serials) and recording_start is None:
                    recording_start = time.monotonic()
                    print("All sensors streaming. Recording started.", flush=True)
                if now - printed >= 1:
                    print(" | ".join(f"{s}: " + (f"{'CONTACT' if latest[s]['depth_nonzero_pixels'] else 'IDLE'}  n={stats[s]['samples']}  depth={latest[s]['depth_peak_sdk']:.4f}  shear={latest[s]['shear_abs_mean_sdk']:.4f}" if s in latest else "CONNECTING") for s in serials), flush=True)
                    printed = now
    except KeyboardInterrupt:
        failure = "interrupted"
    except Exception as error:
        failure = str(error)
    finally:
        stop.set()
        for process in processes:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        report = {"status": "failed" if failure else "completed", "error": failure, "sensors": stats, "elapsed_seconds": time.monotonic() - began, "actuator_commands_sent": 0}
        (args.output / "summary.json").write_text(json.dumps(report, indent=2))
        write_report(args.output, report)
        print(json.dumps(report, indent=2), flush=True)
    if failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
