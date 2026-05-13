#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI entrypoint for TacSE3: tactile sensing → SE(3) motion → UR5.

Default path delegates to ``sensor_loop.simple_sensor_loop_cli`` (contact-mask centroid
translation when ``DEFAULT_CONTACT_CENTER_TRANSLATION`` is True), matching the original
``run_realtime_monitor.py`` behavior.

``--legacy-daimon-monitor`` uses the headless ``DaimonMonitor`` path without centroid-based
translation.

Example::

    pip install -e /path/to/TacSE3
    python -m tacse3.run_monitor --sensor-serial "ID1 ID2" --visualize --duration 100

With optional LeRobot UR5 execution::

    python -m tacse3.run_monitor --sensor-serial "ID1 ID2" --execute --ip 192.168.1.1
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("run_monitor")

from tacse3.sensor_loop import (
    add_primary_cli_args,
    simple_sensor_loop_cli,
    DEFAULT_MIN_CHANGE_DEG,
    DEFAULT_MIN_CHANGE_DEG_X,
    DEFAULT_MIN_CHANGE_DEG_Y,
    DEFAULT_MIN_CHANGE_DEG_Z,
    DEFAULT_ANGLE_SCALE,
    DEFAULT_ANGLE_SCALE_X,
    DEFAULT_ANGLE_SCALE_Y,
    DEFAULT_ANGLE_SCALE_Z,
    DEFAULT_PPMM,
    DEFAULT_CONTACT_THRESH_MM,
    DEFAULT_ROTATION_SCALE,
    DEFAULT_DURATION,
    DEFAULT_INTERVAL,
    DEFAULT_SMOOTH_WINDOW,
    DEFAULT_TOOL_OFFSET_Z,
    DEFAULT_FRAME,
    DEFAULT_MAX_STEP_DEG,
    DEFAULT_AXIS_ORDER,
    DEFAULT_AXIS_SIGN,
    DEFAULT_CONTACT_CENTER_TRANSLATION,
    DEFAULT_CONTACT_CENTER_TRANSLATION_MODE,
)

try:
    from tacse3.core.monitor import DaimonMonitor

    CORE_AVAILABLE = True
except ImportError as e:
    log.error("Cannot import core monitor module: %s", e)
    CORE_AVAILABLE = False

from tacse3.sensor_loop import PoseVisualizer, VISUALIZER_AVAILABLE


def _make_apply_args(rt):
    """Build a namespace compatible with ``simple_sensor_loop_cli`` from this script's parse result."""
    p = argparse.ArgumentParser()
    add_primary_cli_args(p)
    a = p.parse_args([])
    a.simple_loop = True
    a.sensor_serial = rt.sensor_serial
    a.ip = rt.ip
    a.execute = bool(getattr(rt, "execute", False))
    a.visualize = bool(getattr(rt, "visualize", False))
    a.duration = float(rt.duration)
    a.interval = float(rt.interval)
    a.smooth_window = int(rt.smooth_window)
    a.min_change_deg = float(rt.min_change_deg)
    a.min_change_deg_x = float(rt.min_change_deg_x)
    a.min_change_deg_y = float(rt.min_change_deg_y)
    a.min_change_deg_z = float(rt.min_change_deg_z)
    a.shear_scale = float(rt.shear_scale)
    a.angle_scale = float(rt.angle_scale)
    a.angle_scale_x = float(rt.angle_scale_x)
    a.angle_scale_y = float(rt.angle_scale_y)
    a.angle_scale_z = float(rt.angle_scale_z)
    a.tool_offset_z = float(rt.tool_offset_z)
    a.frame = rt.frame
    a.no_invert_rotation = bool(getattr(rt, "no_invert_rotation", False))
    a.baseline_frames = int(rt.baseline_frames)
    a.depth_in_mm = bool(getattr(rt, "depth_in_mm", False))
    a.use_divfree_only = bool(getattr(rt, "use_divfree_only", False))
    a.ppmm = float(rt.ppmm)
    a.contact_thresh_mm = float(rt.contact_thresh_mm)
    a.rotation_scale = float(rt.rotation_scale)
    a.send_delta = bool(getattr(rt, "send_delta", False))
    a.max_step_deg = float(rt.max_step_deg)
    a.axis_order = list(rt.axis_order)
    a.axis_sign = list(rt.axis_sign)
    if getattr(rt, "no_contact_center_translation", False):
        a.contact_center_translation = False
    else:
        a.contact_center_translation = DEFAULT_CONTACT_CENTER_TRANSLATION
    a.contact_center_translation_mode = getattr(
        rt, "contact_center_translation_mode", DEFAULT_CONTACT_CENTER_TRANSLATION_MODE
    )
    return a


def build_parser():
    p = argparse.ArgumentParser(
        description="TacSE3: realtime tactile loop (delegates to sensor_loop by default)."
    )
    p.add_argument("--sensor-serial", type=str, required=True, help="Space or comma separated sensor serial numbers")
    p.add_argument("--ip", type=str, default="192.168.1.1", help="Robot IP address")
    p.add_argument("--execute", action="store_true", help="Execute rotation commands on robot")
    p.add_argument("--visualize", action="store_true", help="Show realtime OpenCV + optional PoseVisualizer (legacy path)")
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="Loop duration in seconds (0=forever)")
    p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL, help="Sleep between iterations (seconds)")
    p.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW, help="Dual-window half-size (buffer = 2*N)")
    p.add_argument("--min-change-deg", type=float, default=DEFAULT_MIN_CHANGE_DEG, help="Fallback min angular change (deg) for all axes")
    p.add_argument("--min-change-deg-x", type=float, default=DEFAULT_MIN_CHANGE_DEG_X, help="Min angular change X (deg)")
    p.add_argument("--min-change-deg-y", type=float, default=DEFAULT_MIN_CHANGE_DEG_Y, help="Min angular change Y (deg)")
    p.add_argument("--min-change-deg-z", type=float, default=DEFAULT_MIN_CHANGE_DEG_Z, help="Min angular change Z (deg)")
    p.add_argument("--shear-scale", type=float, default=1.0, help="Scale factor for shear data")
    p.add_argument("--angle-scale", type=float, default=DEFAULT_ANGLE_SCALE, help="Fallback angle gain for all axes")
    p.add_argument("--angle-scale-x", type=float, default=DEFAULT_ANGLE_SCALE_X, help="Gain for rotation about X")
    p.add_argument("--angle-scale-y", type=float, default=DEFAULT_ANGLE_SCALE_Y, help="Gain for rotation about Y")
    p.add_argument("--angle-scale-z", type=float, default=DEFAULT_ANGLE_SCALE_Z, help="Gain for rotation about Z")
    p.add_argument("--tool-offset-z", type=float, default=DEFAULT_TOOL_OFFSET_Z, help="TCP to tooltip offset (meters)")
    p.add_argument("--frame", choices=["global", "local"], default=DEFAULT_FRAME, help="Increment frame: base or tool")
    p.add_argument(
        "--no-invert-rotation",
        action="store_true",
        help="Send estimator direction without negation (default negates for opposite TCP motion)",
    )
    p.add_argument("--baseline-frames", type=int, default=0, help="Frames to average for depth baseline")
    p.add_argument("--depth-in-mm", action="store_true", help="Treat depth as millimeters")
    p.add_argument("--use-divfree-only", action="store_true", help="Helmholtz div-free shear component only")
    p.add_argument("--ppmm", type=float, default=DEFAULT_PPMM, help="Pixels per millimeter")
    p.add_argument("--contact-thresh-mm", type=float, default=DEFAULT_CONTACT_THRESH_MM, help="Contact mask depth threshold (mm)")
    p.add_argument("--rotation-scale", type=float, default=DEFAULT_ROTATION_SCALE, help="delta_scale / rotvec gain before send_action")
    p.add_argument(
        "--no-send-swap-xz",
        action="store_true",
        help="[legacy DaimonMonitor only] Disable X/Z swap on the vector sent to the robot",
    )
    p.add_argument(
        "--send-delta",
        action="store_true",
        help="Use windowed delta (avg_last - avg_first) as rotation command; else smoothed absolute",
    )
    p.add_argument("--max-step-deg", type=float, default=DEFAULT_MAX_STEP_DEG, help="Clamp rotation-vector norm per step (deg)")
    p.add_argument("--axis-order", nargs=3, type=int, default=DEFAULT_AXIS_ORDER, metavar=("I0", "I1", "I2"))
    p.add_argument("--axis-sign", nargs=3, type=int, default=DEFAULT_AXIS_SIGN, metavar=("S0", "S1", "S2"))
    p.add_argument(
        "--no-contact-center-translation",
        action="store_true",
        help="Disable contact-mask centroid translation; use joint tx,ty from shear estimator instead",
    )
    p.add_argument(
        "--contact-center-translation-mode",
        choices=["consecutive", "window"],
        default=DEFAULT_CONTACT_CENTER_TRANSLATION_MODE,
        help="How centroid displacement is computed (window uses --smooth-window)",
    )
    p.add_argument(
        "--legacy-daimon-monitor",
        action="store_true",
        help="Use headless DaimonMonitor + optional custom visualize loop (no centroid translation)",
    )
    return p


def _run_legacy_daimon_monitor(args, robot):
    """Previous behavior: DaimonMonitor only (no contact-centroid path)."""
    min_change_deg_xyz = [
        float(args.min_change_deg_x),
        float(args.min_change_deg_y),
        float(args.min_change_deg_z),
    ]
    angle_scale_xyz = [
        float(args.angle_scale_x),
        float(args.angle_scale_y),
        float(args.angle_scale_z),
    ]
    log.info("Initializing DaimonMonitor (legacy) with sensors: %s", args.sensor_serial)
    serials = [s.strip() for s in args.sensor_serial.replace(",", " ").split() if s.strip()]
    monitor = DaimonMonitor(
        sensor_serials=serials,
        robot=robot,
        ppmm=args.ppmm,
        shear_scale=args.shear_scale,
        angle_scale=angle_scale_xyz,
        smooth_window=args.smooth_window,
        min_change_deg=min_change_deg_xyz,
        baseline_frames=args.baseline_frames,
        depth_in_mm=args.depth_in_mm,
        use_divfree_only=args.use_divfree_only,
        interval=args.interval,
        execute=args.execute,
        tool_offset_z=args.tool_offset_z,
        frame=args.frame,
        invert_rotation=not args.no_invert_rotation,
        send_swap_xz=not args.no_send_swap_xz,
        contact_thresh_mm=args.contact_thresh_mm,
        delta_scale=args.rotation_scale,
        send_delta=args.send_delta,
        axis_order=list(args.axis_order),
        axis_sign=list(args.axis_sign),
        max_step_deg=args.max_step_deg,
        include_translation=True,
    )
    log.info("Running DaimonMonitor loop for %s s...", args.duration if args.duration > 0 else "unlimited")
    if getattr(args, "visualize", False) and VISUALIZER_AVAILABLE:
        pv = PoseVisualizer(window_name="Realtime Pose", min_change_deg=min_change_deg_xyz)
        try:
            end_time = time.time() + args.duration if args.duration > 0 else None
            fps_start = time.perf_counter()
            fps_count = 0
            while True:
                if end_time and time.time() >= end_time:
                    break
                res = monitor.step()
                if res and "omega_rad" in res:
                    omega = res["omega_rad"]
                    cumulative_deg = np.degrees(np.asarray(res.get("cumulative_omega_rad", np.zeros(3)), dtype=float))
                    print(
                        "\rCumulative rotation (deg) [rx, ry, rz]: " f"{np.round(cumulative_deg, 4)}",
                        end="",
                        flush=True,
                    )
                    pose = [0.0, 0.0, 0.0, float(omega[0]), float(omega[1]), float(omega[2])]
                    pv.update(
                        pose,
                        accumulate=not np.allclose(omega, 0.0),
                        delta_rotation="rotvec",
                        accumulate_frame=args.frame,
                    )
                fps_count += 1
                if fps_count % 30 == 0:
                    elapsed = time.perf_counter() - fps_start
                    log.info(
                        "[FPS] frames=%s elapsed=%.2fs fps=%.2f",
                        fps_count,
                        elapsed,
                        fps_count / max(elapsed, 1e-6),
                    )
                time.sleep(args.interval)
        except KeyboardInterrupt:
            log.info("Visual loop interrupted")
        finally:
            try:
                pv.close()
            except Exception:
                pass
    else:
        monitor.loop(duration=args.duration)
    monitor.shutdown()


def main():
    p = build_parser()
    args = p.parse_args()
    min_change_deg_xyz = [float(args.min_change_deg_x), float(args.min_change_deg_y), float(args.min_change_deg_z)]
    angle_scale_xyz = [float(args.angle_scale_x), float(args.angle_scale_y), float(args.angle_scale_z)]

    if not CORE_AVAILABLE:
        if getattr(args, "visualize", False) and VISUALIZER_AVAILABLE:
            log.info("Core not available; running simulated visualizer loop.")
            pv = PoseVisualizer(window_name="Simulated Realtime Pose", min_change_deg=min_change_deg_xyz)
            try:
                end_time = time.time() + args.duration if args.duration > 0 else None
                t0 = time.time()
                cumulative_angle_deg = np.zeros(3, dtype=float)
                while True:
                    if end_time and time.time() >= end_time:
                        break
                    t = time.time() - t0
                    omega = np.array(
                        [
                            0.04 * np.sin(2.0 * np.pi * 0.5 * t),
                            0.03 * np.sin(2.0 * np.pi * 0.35 * t + 0.7),
                            0.06 * np.sin(2.0 * np.pi * 0.25 * t + 1.3),
                        ],
                        dtype=float,
                    )
                    omega = omega * np.asarray(angle_scale_xyz, dtype=float)
                    omega_deg = np.degrees(omega)
                    omega_deg[np.abs(omega_deg) < np.asarray(min_change_deg_xyz, dtype=float)] = 0.0
                    if not args.no_invert_rotation:
                        omega = -np.asarray(omega, dtype=float)
                    cumulative_angle_deg += np.degrees(omega)
                    print(
                        "\rCumulative rotation (deg) [rx, ry, rz]: " f"{np.round(cumulative_angle_deg, 4)}",
                        end="",
                        flush=True,
                    )
                    pose = [0.0, 0.0, 0.0, float(omega[0]), float(omega[1]), float(omega[2])]
                    pv.update(
                        pose,
                        accumulate=not np.allclose(omega, 0.0),
                        delta_rotation="rotvec",
                        accumulate_frame=args.frame,
                    )
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                log.info("Simulated visual loop interrupted")
            finally:
                try:
                    pv.close()
                except Exception:
                    pass
            return
        sys.exit(
            "Please confirm dmrobotics and tacse3.core are properly installed "
            "(or run with --visualize to use simulated UI)."
        )

    robot = None
    if args.execute:
        try:
            from lerobot.robots.ur5.UR5Robot import UR5Robot
            from lerobot.robots.ur5.UR5Config import UR5Config

            try:
                cfg = UR5Config(robot_ip=args.ip, use_cameras=False, cam_ids=[])
            except TypeError:
                cfg = UR5Config(robot_ip=args.ip)
            robot = UR5Robot(cfg)
        except Exception as e:
            log.warning("UR5 not available (%s); continuing without on-robot execution.", e)
            robot = None
        else:
            log.info("Robot object created for %s (connection may occur in worker thread).", args.ip)
    else:
        log.info("Dry-run: no robot commands unless --execute.")

    try:
        if getattr(args, "legacy_daimon_monitor", False):
            _run_legacy_daimon_monitor(args, robot)
        else:
            apply_args = _make_apply_args(args)
            log.info(
                "Using sensor_loop.simple_sensor_loop_cli (contact_center_translation=%s, mode=%s).",
                apply_args.contact_center_translation,
                apply_args.contact_center_translation_mode,
            )
            simple_sensor_loop_cli(apply_args, robot)
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    except Exception as e:
        log.error("Monitor failed: %s", e, exc_info=True)
    finally:
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
