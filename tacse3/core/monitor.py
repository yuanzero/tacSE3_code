import time
import logging
import numpy as np
import concurrent.futures
import importlib
import sys
from pathlib import Path

# Try imports
_sensor_import_error = None
try:
    from dmrobotics import Sensor
except Exception as e:
    Sensor = None
    _sensor_import_error = e
    try:
        # When running scripts directly, the repo `src/` directory may not have
        # won import precedence; retry explicitly from the local source tree.
        src_root = Path(__file__).resolve().parents[2]
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
        Sensor = importlib.import_module("dmrobotics").Sensor
        _sensor_import_error = None
    except Exception as retry_error:
        Sensor = None
        _sensor_import_error = retry_error

from .math_utils import rotvec_to_matrix, clamp_rotation_step
from .sensor_processing import build_contact_from_daimon
from .estimator import estimate_rotation_from_shear
from .fusion import fuse_rotations
from .units import DEFAULT_PPMM
from .robot_worker import RobotWorker

log = logging.getLogger("daimon_monitor")


class DaimonMonitor:
    """
    Main controller class for monitoring Daimon sensors, estimating rotation,
    fusing data, and commanding a robot via RobotWorker.

    Rotation post-processing matches ``apply_ur5_rotation.py`` legacy
    ``--simple-loop`` / ``--visualize`` path: fuse → axis remap → per-axis gain →
    dual-window smooth (buffer ``2 * smooth_window``) → optional send_delta →
    min-change masking → norm clamp → invert → robot X/Z swap.
    """
    def __init__(self, 
                 sensor_serials: list, 
                 robot=None,
                 ppmm=DEFAULT_PPMM,
                 shear_scale=1.0,
                 angle_scale=1.0, 
                 smooth_window=10,
                 min_change_deg=8.5,
                 baseline_frames=0,
                 depth_in_mm=True,
                 use_divfree_only=False,
                 interval=0.05,
                 execute=False,
                 tool_offset_z=0.0,
                 frame='local',
                 invert_rotation=False,
                 send_swap_xz=False,
                 contact_thresh_mm=0.5,
                 delta_scale=1.0,
                 send_delta=False,
                 axis_order=None,
                 axis_sign=None,
                 fusion_signs=None,
                 max_step_deg=3.0,
                 include_translation=True):
        
        self.sensor_serials = sensor_serials
        self.robot = robot
        self.ppmm = ppmm
        self.shear_scale = shear_scale
        self.angle_scale = self._normalize_3vec(angle_scale, name="angle_scale")
        self.smooth_window = int(smooth_window) if smooth_window else 0
        self.min_change_deg = self._normalize_3vec(min_change_deg, name="min_change_deg")
        self.baseline_frames = baseline_frames
        self.depth_in_mm = depth_in_mm
        self.use_divfree_only = bool(use_divfree_only)
        self.interval = interval
        self.execute = execute
        self.tool_offset_z = tool_offset_z
        self.frame = frame if frame in ('global', 'local') else 'local'
        self.invert_rotation = bool(invert_rotation)
        self.send_swap_xz = bool(send_swap_xz)
        self.contact_thresh_mm = float(contact_thresh_mm)
        self.delta_scale = float(delta_scale)
        self.send_delta = bool(send_delta)
        self.fusion_signs = None if fusion_signs is None else np.asarray(fusion_signs, dtype=float)
        self.max_step_deg = float(max_step_deg)
        self.include_translation = bool(include_translation)

        ao = [0, 1, 2] if axis_order is None else [int(x) for x in axis_order]
        sg = [1, 1, 1] if axis_sign is None else [int(x) for x in axis_sign]
        if len(ao) != 3 or len(sg) != 3:
            raise ValueError("axis_order and axis_sign must have length 3")
        self.axis_order = ao
        self.axis_sign = sg
        
        self.sensors = []
        self.baselines = []
        self.worker = None
        self.running = False
        # latest incremental rotation command (rad) shared by robot + visualizer
        self.last_omega_rad = None
        self.last_omega_time = None
        self.last_raw_omega_rad = None
        self.cumulative_omega_rad = np.zeros(3, dtype=float)
        
        # Dual-window buffer: same semantics as apply_ur5_rotation legacy loop
        self.rot_buffer = []
        self._sw = max(1, self.smooth_window) if self.smooth_window > 0 else 1
        self.buffer_size = 2 * self._sw

        # Initialize sensors
        self._init_sensors()
        
        # Initialize baselines if requested
        if self.baseline_frames > 0:
            self._acquire_baselines()
            
        # Initialize robot worker
        if self.execute and self.robot:
            self.worker = RobotWorker(self.robot)
            self.worker.start()

    @staticmethod
    def _normalize_3vec(value, name="value"):
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 1:
            arr = np.repeat(arr, 3)
        elif arr.size != 3:
            raise ValueError(f"{name} must be a scalar or length-3 iterable")
        return arr

    def _remap_axes(self, omega_rad):
        """Sensor-frame → robot-frame (same as apply_ur5_rotation legacy axis_order / axis_sign)."""
        o = np.asarray(omega_rad, dtype=float)
        return np.array([o[i] for i in self.axis_order], dtype=float) * np.asarray(self.axis_sign, dtype=float)

    def _init_sensors(self):
        if not Sensor:
            log.error(
                "dmrobotics not installed/importable under Python %s (%s): %s",
                sys.version.split()[0],
                sys.executable,
                _sensor_import_error,
            )
            return
        log.info(f"Initializing {len(self.sensor_serials)} sensors: {self.sensor_serials}")
        for i, ser in enumerate(self.sensor_serials):
            try:
                log.info(f"Connecting sensor {i} ({ser}) with timeout...")
                # initialize sensor creation/connect in a worker to avoid blocking
                def _create_and_connect(sid):
                    obj = Sensor(sid)
                    if hasattr(obj, 'connect'):
                        obj.connect()
                    return obj

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(_create_and_connect, ser)
                    try:
                        s_obj = fut.result(timeout=3.0)
                        self.sensors.append(s_obj)
                        log.info(f"Connected sensor {i} ({ser}).")
                    except concurrent.futures.TimeoutError:
                        log.warning(f"Timeout initializing sensor {ser}; skipping for now.")
                    except Exception as e:
                        log.warning(f"Failed to initialize sensor {ser}: {e}")
                        # don't append this sensor
            except Exception as e:
                log.warning(f"Failed to initialize sensor {ser}: {e}")
        
        if not self.sensors:
             log.error("No sensors could be initialized.")

    def _acquire_baselines(self):
        if self.baseline_frames <= 0:
            return

        self.baselines = [None] * len(self.sensors)
        log.info(f"Acquiring {self.baseline_frames} baseline frames for {len(self.sensors)} sensors...")
        for i, sens in enumerate(self.sensors):
            try:
                log.info(f"Acquiring baseline for sensor {i}...")
                depths = []
                for f in range(self.baseline_frames):
                    d = np.array(sens.getDepth(), dtype=np.float32)
                    depths.append(d)
                    if f % 10 == 0:
                         log.info(f"Sensor {i} baseline frame {f}/{self.baseline_frames}")
                
                # Convert to mm if needed for baseline storage (or keep consistent with build_contact)
                mean_depth = np.mean(np.stack(depths, axis=0), axis=0)
                if not self.depth_in_mm:
                    mean_depth *= 1000.0  # Store in mm
                self.baselines[i] = mean_depth
                log.info(f"Baseline acquired for sensor {i}.")
            except Exception as e:
                log.warning(f"Baseline acquisition failed for sensor {i}: {e}")

    def _read_depth_shear(self, sensor, timeout=0.5):
        """Read depth and shear from a sensor with a timeout to avoid blocking the loop.

        Returns (depth, shear) or (None, None) on timeout/error.
        """
        try:
            def _read(s):
                return s.getDepth(), s.getShear()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_read, sensor)
                return fut.result(timeout=timeout)
        except Exception as e:
            log.debug(f"Sensor read timeout/error: {e}")
            return None, None

    def loop(self, duration=0.0):
        """
        Run the main monitoring loop.
        Args:
            duration: Time in seconds to run. 0 for infinite.
        """
        end_time = time.time() + duration if duration > 0 else None
        self.running = True

        log.info("Starting monitoring loop...")
        try:
            loop_count = 0
            while self.running:
                if end_time and time.time() >= end_time:
                    break
                
                t_start = time.time()
                
                # Check execution rate
                if loop_count % 100 == 0:
                    log.debug(f"Loop running... count={loop_count}")

                try:
                    result = self.step()
                except Exception as e:
                    # Catch all exceptions in step to prevent loop crash
                    log.warning(f"Error in monitor step: {e}")
                    result = None

                if not result:
                    # Headless mode has no stdout print; INFO here so the process is not silent.
                    if loop_count % 200 == 0:
                        log.info(
                            "No contact or valid sensor data yet (empty mask / read timeout). "
                            "Loop is running."
                        )
                
                # Sleep rest of interval
                elapsed = time.time() - t_start
                rest = max(0.001, self.interval - elapsed)
                time.sleep(rest)
                loop_count += 1
                
        except KeyboardInterrupt:
            log.info("Loop interrupted by user.")
        finally:
            self.shutdown()

    def step(self):
        """Process one sensor frame; same rotation math as apply_ur5_rotation legacy loop."""
        result = self._process_sensors()
        if not result:
            self.rot_buffer = []
            return None

        raw_fused_rad = np.asarray(result['fused_rad'], dtype=float)
        omega_scaled = self._remap_axes(raw_fused_rad) * self.angle_scale

        self.rot_buffer.append(omega_scaled)
        if len(self.rot_buffer) > self.buffer_size:
            self.rot_buffer.pop(0)

        delta_omega = np.zeros(3, dtype=float)
        sw = self._sw
        if len(self.rot_buffer) >= self.buffer_size:
            first_half = self.rot_buffer[:sw]
            last_half = self.rot_buffer[-sw:]
            avg_first = np.mean(first_half, axis=0)
            avg_last = np.mean(last_half, axis=0)
            delta_omega = avg_last - avg_first
            omega_rad_smoothed = avg_last
            if self.send_delta:
                to_send_rad = np.array(delta_omega, dtype=float)
            else:
                to_send_rad = np.array(omega_rad_smoothed, dtype=float)
        else:
            if len(self.rot_buffer) > 0:
                to_send_rad = np.mean(self.rot_buffer, axis=0)
            else:
                to_send_rad = np.zeros(3, dtype=float)

        try:
            change_source = delta_omega if len(self.rot_buffer) >= self.buffer_size else to_send_rad
            change_deg_xyz = np.abs(np.degrees(np.asarray(change_source, dtype=float)))
            zero_mask = change_deg_xyz < self.min_change_deg
            if np.any(zero_mask):
                to_send_rad = np.asarray(to_send_rad, dtype=float)
                to_send_rad[zero_mask] = 0.0
        except Exception:
            pass

        to_send_rad = clamp_rotation_step(to_send_rad, self.max_step_deg)

        if self.invert_rotation:
            to_send_rad = -np.asarray(to_send_rad, dtype=float)

        try:
            self.last_omega_rad = np.array(to_send_rad, dtype=float)
            self.last_omega_time = time.time()
            self.cumulative_omega_rad += to_send_rad
        except Exception:
            pass

        to_send_robot_rad = np.array(to_send_rad, dtype=float)
        if self.send_swap_xz:
            to_send_robot_rad = to_send_robot_rad[[2, 1, 0]]

        if self.worker and not np.allclose(to_send_robot_rad, 0.0):
            req = {
                'multiply': 'left',
                'frame': self.frame,
                'tool_offset_z': self.tool_offset_z,
                'send_delta': self.send_delta,
                'delta_scale': self.delta_scale,
                'max_step_deg': self.max_step_deg,
            }
            if self.send_delta:
                req['omega_rad'] = to_send_robot_rad
            else:
                rvec_scaled = np.asarray(to_send_robot_rad, dtype=float) * float(self.delta_scale)
                req['R'] = rotvec_to_matrix(rvec_scaled)
                req['omega_rad'] = to_send_robot_rad
            self.worker.send(req)

        return {
            'omega_rad': np.array(to_send_rad, dtype=float),
            'raw_omega_rad': raw_fused_rad,
            'cumulative_omega_rad': np.array(self.cumulative_omega_rad, dtype=float),
        }

    def _process_sensors(self):
        """Read all sensors, estimate individual rotations, fuse them."""
        all_rotations_rad = []
        valid_data = False

        for i, s in enumerate(self.sensors):
            try:
                # Guarded read so a resetting/hung sensor can't block the whole loop
                depth, shear = self._read_depth_shear(s, timeout=max(0.1, self.interval * 2.0))
                if depth is None or shear is None:
                    # Skip this sensor for this iteration
                    continue

                # Use hardware rotation if available
                sensor_rot = None
                if hasattr(s, 'getRotation'):
                    try:
                        sensor_rot = s.getRotation()
                    except Exception:
                        sensor_rot = None

                # Build contact (prefer same helper as apply_ur5_rotation legacy loop)
                baseline = self.baselines[i] if i < len(self.baselines) else None
                depth_m, shear_scaled, C_mask = build_contact_from_daimon(
                    depth,
                    shear,
                    self.ppmm,
                    depth_in_mm=self.depth_in_mm,
                    baseline_depth_mm=baseline,
                    press_increases_depth=True,
                    shear_scale=1.0,
                    contact_thresh_mm=self.contact_thresh_mm,
                )

                if not np.any(C_mask):
                    continue

                # Estimate (shear_scale here; not in contact — matches legacy)
                res = estimate_rotation_from_shear(
                    depth_m,
                    shear_scaled,
                    ppmm=self.ppmm,
                    contact_mask=C_mask,
                    shear_scale=float(self.shear_scale),
                    include_translation=self.include_translation,
                    use_divfree_only=self.use_divfree_only,
                )

                omega_rad_one = np.array(res['omega_rad'], dtype=float)

                # Hardware override
                if sensor_rot is not None:
                    try:
                        sensor_rot_arr = np.array(sensor_rot, dtype=float)
                        if sensor_rot_arr.shape == (3,):
                            omega_rad_one = sensor_rot_arr
                    except Exception:
                        pass

                all_rotations_rad.append(omega_rad_one)
                valid_data = True

            except Exception as e:
                log.debug(f"Sensor {i} processing error: {e}")
                # Continue with next sensor
                continue

        if not valid_data:
            return None

        # Fuse (same two-sensor signs as apply_ur5_rotation legacy / fusion.fuse_rotations)
        fused_rad = np.asarray(
            fuse_rotations(all_rotations_rad, fusion_signs=self.fusion_signs),
            dtype=float,
        )
        try:
            self.last_raw_omega_rad = np.array(fused_rad, dtype=float)
        except Exception:
            pass

        return {'fused_rad': fused_rad}

    def get_last_omega_rad(self):
        """Return the most recent incremental rotation command (rad), or None if unavailable."""
        return self.last_omega_rad

    def get_last_omega_deg(self):
        """Return the most recent incremental rotation command (deg), or None if unavailable."""
        if self.last_omega_rad is None:
            return None
        try:
            return np.degrees(self.last_omega_rad).tolist()
        except Exception:
            return None

    def get_cumulative_omega_rad(self):
        """Return the accumulated commanded rotation (rad)."""
        return np.array(self.cumulative_omega_rad, dtype=float)

    def get_cumulative_omega_deg(self):
        """Return the accumulated commanded rotation (deg)."""
        try:
            return np.degrees(self.cumulative_omega_rad).tolist()
        except Exception:
            return None

    def shutdown(self):
        self.running = False
        if self.worker:
            self.worker.stop()
            self.worker.join()
