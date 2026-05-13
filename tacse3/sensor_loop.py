"""TacSE3 — Daimon-style tactile sensors → SE(3) motion → UR5 commands.

Extracted from ``realtime_daimon_rot.apply_ur5_rotation`` into this package.
Public API: ``add_primary_cli_args``, ``simple_sensor_loop_cli``, and module-level defaults.
"""
import argparse
import concurrent.futures
import logging
import threading
import time
from collections import deque

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from .contact_center_translation import ContactCenterTranslationState
from .core.monitor import DaimonMonitor
from .core.robot_control import compute_action_from_pose
from .core.math_utils import rotvec_to_matrix, euler_angles_to_rotation_matrix
from .core.estimator import estimate_rotation_from_shear, pixel_xy_grid
from .core.sensor_processing import build_contact_from_daimon

try:
    from .pose_visualizer import PoseVisualizer
except Exception:
    PoseVisualizer = None
VISUALIZER_AVAILABLE = PoseVisualizer is not None

CORE_AVAILABLE = True

def estimate_twist_from_depth_variation(points_xy_or_xyz, u, v, depth_t, depth_prev, dt):
    """
    Estimate the rigid-body twist [omega; t] from measured point velocities where
    the normal velocity w is computed from temporal depth variation.

    Theory (see notes): w_i = (z_i(t) - z_i(t-dt)) / dt and
    v_i = omega x p_i + t, with p_i = (x_i, y_i, z_i).

    Inputs:
      - points_xy_or_xyz: (N,2) or (N,3) array. If (N,2), treated as (x,y) and z is taken from depth_t.
      - u, v: (N,) arrays of point tangential velocities (units consistent with depth; e.g. mm/sec)
      - depth_t: (N,) depths at time t (same units as shear/depth_prev, e.g. mm)
      - depth_prev: (N,) depths at time t-dt (same units as depth_t)
      - dt: scalar time difference (seconds)

    Returns:
      dict with keys:
        - 'omega': (3,) angular velocity vector (rad/s)
        - 't': (3,) linear translation vector (same length unit/s as depth, e.g. mm/s)
        - 'xi': (6,) stacked [omega; t]
        - 'residuals': residual norm from least-squares (sum of squared residuals)
        - 'A': the design matrix (3N x 6)
        - 'b': the stacked velocity vector (3N,)

    Notes:
      - The linear system solved is: A xi = b, where for each point i:
          [  0   z_i  -y_i  1 0 0 ] [omega_x]   [ u_i ]
          [ -z_i  0   x_i  0 1 0 ] [omega_y] = [ v_i ]
          [  y_i -x_i   0  00 1 ] [omega_z]   [ w_i ]
      - Requires at least two non-collinear points for a well-conditioned estimate.
    """
    pts = np.asarray(points_xy_or_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError('points_xy_or_xyz must be shape (N,2) or (N,3)')

    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    depth_t = np.asarray(depth_t, dtype=float).ravel()
    depth_prev = np.asarray(depth_prev, dtype=float).ravel()

    # NOTE: depth and shear are assumed to be provided in the sensor's native
    # units (e.g. mm) and outputs will be expressed in those units per second
    # (e.g. mm/s for translation).

    if not (u.size == v.size == depth_t.size == depth_prev.size == pts.shape[0]):
        raise ValueError('Input arrays must have matching first dimension N')

    # build full 3D points: override or append z from depth_t
    if pts.shape[1] == 2:
        x = pts[:, 0]
        y = pts[:, 1]
        z = depth_t
    else:
        x = pts[:, 0]
        y = pts[:, 1]
        # override provided z with depth_t to treat z displacement as depth
        z = depth_t

    # normal velocity from depth change
    if dt == 0:
        raise ValueError('dt must be non-zero')
    w = (depth_t - depth_prev) / float(dt)

    N = x.size
    # Build A (3N x 6) and b (3N)
    A = np.zeros((3 * N, 6), dtype=float)
    b = np.zeros(3 * N, dtype=float)
    for i in range(N):
        xi = x[i]
        yi = y[i]
        zi = z[i]
        row0 = 3 * i
        row1 = row0 + 1
        row2 = row0 + 2
        # u row
        A[row0, 0] = 0.0
        A[row0, 1] = zi
        A[row0, 2] = -yi
        A[row0, 3] = 1.0
        A[row0, 4] = 0.0
        A[row0, 5] = 0.0
        b[row0] = u[i]
        # v row
        A[row1, 0] = -zi
        A[row1, 1] = 0.0
        A[row1, 2] = xi
        A[row1, 3] = 0.0
        A[row1, 4] = 1.0
        A[row1, 5] = 0.0
        b[row1] = v[i]
        # w row
        A[row2, 0] = yi
        A[row2, 1] = -xi
        A[row2, 2] = 0.0
        A[row2, 3] = 0.0
        A[row2, 4] = 0.0
        A[row2, 5] = 1.0
        b[row2] = w[i]

    # Solve least squares
    xi_sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    omega = xi_sol[:3]
    t_vec = xi_sol[3:6]
    res_norm = float(residuals.sum()) if residuals.size > 0 else float(np.sum((A.dot(xi_sol) - b) ** 2))

    return {
        'omega': omega,
        't': t_vec,
        'xi': xi_sol,
        'residuals': res_norm,
        'A': A,
        'b': b,
        'rank': rank,
        'singular_values': s,
    }


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ------------------- User-visible defaults (tweak these at top of file) -------------------
# 机器人默认IP地址
DEFAULT_ROBOT_IP = "192.168.1.1"
# 是否执行动作（True为发送指令给机器人，False为仅打印/模拟）
DEFAULT_EXECUTE = False
# 是否可视化（显示OpenCV窗口）
# NOTE: default False — previously True caused the GUI to be enabled by default
# which made the script run visualization even when the user did not request it.
DEFAULT_VISUALIZE = True
# 是否运行简单的传感器循环
# NOTE: default False — previously True caused the realtime sensor loop to start
# automatically and block if sensors were resetting. Make explicit via flags.
DEFAULT_SIMPLE_LOOP = True
# 是否使用传感器输入
# NOTE: default False — previously True forced sensor initialization on launch.
DEFAULT_USE_SENSOR = False
# 传感器序列号（请在此处填入您的传感器ID，例如 "M2505150016 M2505150299"）
DEFAULT_SENSOR_SERIAL = "M2505150016 M2505150299"
DEFAULT_SERIAL = None  # legacy alias for default serial resolution in loops
# 是否保存可视化帧到磁盘
DEFAULT_SAVE_FRAMES = False
# 是否发送增量旋转（而不是绝对旋转）

# 增量/发送旋转的缩放比例（重命名为 DEFAULT_ROTATION_SCALE）
DEFAULT_ROTATION_SCALE = 0.5

# 兼容旧名
DEFAULT_DELTA_SCALE = DEFAULT_ROTATION_SCALE
# 增量旋转/发送模式：
# True = 发送窗口差分作为增量（默认行为）；
# False = 发送平滑后的原始旋转（smoothed raw）。
DEFAULT_SEND_DELTA = False
# 循环持续时间（秒），55.0秒
DEFAULT_DURATION = 2550.0
# 循环间隔时间（秒）
DEFAULT_INTERVAL = 0.05

# ------------------- Axis remapping (input-side, sensor -> robot) -------------------
# Default rotation mapping (DEFAULT_AXIS_ORDER = [0, 1, 2]):
# robot X <- sensor X, robot Y <- sensor Y, robot Z <- sensor Z
# i.e. robot_axes = [sensor[0], sensor[1], sensor[2]]
DEFAULT_AXIS_ORDER = [0, 1, 2]
# Per-axis sign flips (+1 or -1)
DEFAULT_AXIS_SIGN = [1, 1, 1]
# Translation uses independent remap by default (identity) to avoid unexpected axis swapping.
DEFAULT_TRANSLATION_AXIS_ORDER = [0, 1, 2]
DEFAULT_TRANSLATION_AXIS_SIGN = [1, 1, 1]

# Coordinate frame and Tool parameters
# 旋转参考系：'global'（基座坐标系）或 'local'（工具/末端执行器坐标系）
DEFAULT_FRAME = 'local'          # 'global' (base) or 'local' (tool/end-effector)
# 与传感器估计的角速度方向相反再下发（末端在工具系下运动时常见）；关闭请用 --no-invert-rotation
DEFAULT_NO_INVERT_ROTATION = False  # False => 默认对 omega / R 取反
# 工具Z轴偏移量（从法兰到工具尖端的距离，单位：米）
DEFAULT_TOOL_OFFSET_Z = 0.25
# 矩阵乘法顺序（'left' 或 'right'，遗留参数）
DEFAULT_MULTIPLY = 'left'         # Legacy matrix multiplication order

# Algorithm parameters
# 像素每毫米（传感器分辨率相关参数）
DEFAULT_PPMM = 0.0634
# 剪切力缩放比例 (增大此值会增强剪切力信号，从而增大估算的旋转)
DEFAULT_SHEAR_SCALE = 1.0

# 基准帧数（用于校准归零）
DEFAULT_BASELINE_FRAMES = 0
# 深度数据是否以毫米为单位
DEFAULT_DEPTH_IN_MM = False
# 接触阈值（毫米）
# Must match the contact mask logic in ``build_contact_from_daimon`` (legacy loop always used its default 0.5 mm).
DEFAULT_CONTACT_THRESH_MM = 0.5
# 平滑处理：计算前后N帧的平均值
DEFAULT_SMOOTH_WINDOW = 2

# Estimation parameters
# 是否估算平移（区分平移和旋转，如果为False则假设纯旋转）
DEFAULT_INCLUDE_TRANSLATION = True
# True：平面位移来自接触掩膜中心轨迹（contact_center_translation 模块）；False：沿用剪切/深度 twist 估计的平移
DEFAULT_CONTACT_CENTER_TRANSLATION = True
# 接触中心位移模式：与 ContactCenterTranslationState 一致（window 使用下面的 smooth_window）
DEFAULT_CONTACT_CENTER_TRANSLATION_MODE = 'consecutive'
# 是否使用亥姆霍兹分解仅保留无散分量（去除挤压造成的形变干扰）目前效果不好，默认关闭
DEFAULT_USE_DIVFREE_ONLY = False
# 是否使用基于深度时间变化的估计器（使用深度差计算法线速度w）目前效果不好，默认关闭
DEFAULT_USE_DEPTH_VARIATION = False

# Safety parameters
DEFAULT_WARMUP_FRAMES = 10        # Ignore first N valid frames
DEFAULT_MAX_STEP_DEG = 3.0        # Max degrees rotation per interval




# 平移增量阈值（米）：针对经过 translation_scale 后的每步位移增量，
# 范数小于该值则忽略，抑制微小抖动累计
DEFAULT_MIN_TRANSLATION_DELTA_M = 0.5
# 三轴平移增量阈值（米）：针对经过 translation_scale 后的每步位移增量，
# 按轴过滤微小平移，未单独设置时回退到共享阈值
# DEFAULT_MIN_TRANSLATION_DELTA_M_X = 0.001
# DEFAULT_MIN_TRANSLATION_DELTA_M_Y = 0.001
# DEFAULT_MIN_TRANSLATION_DELTA_M_Z = 0.001
DEFAULT_MIN_TRANSLATION_DELTA_M_X = 0.00
DEFAULT_MIN_TRANSLATION_DELTA_M_Y = 0.00
DEFAULT_MIN_TRANSLATION_DELTA_M_Z = 0.00
# 平移增益（幅度缩放）：用于控制输出平移的整体和各轴大小
DEFAULT_TRANSLATION_SCALE = 1.0
DEFAULT_TRANSLATION_SCALE_X = 0.0075
DEFAULT_TRANSLATION_SCALE_Y = 0.01
DEFAULT_TRANSLATION_SCALE_Z = 1

# 角度缩放比例 (如果觉得旋转幅度太小，请增大此值，例如 2.0 或 5.0)

DEFAULT_ANGLE_SCALE = 1.0
DEFAULT_ANGLE_SCALE_X = 0.25
DEFAULT_ANGLE_SCALE_Y = 1
DEFAULT_ANGLE_SCALE_Z = 15

# Minimum change (degrees) between smoothed windows. Below this threshold the
# command omega is treated as zero to avoid tiny jitter when there is contact
# but almost no real angular motion.
DEFAULT_MIN_CHANGE_DEG = 5
DEFAULT_MIN_CHANGE_DEG_X = 2
DEFAULT_MIN_CHANGE_DEG_Y = 5
DEFAULT_MIN_CHANGE_DEG_Z = 5
DEFAULT_VIZ_AXIS_MAPPING = ("-x", "z", "-y")

# Input / Debug
DEFAULT_MATRIX = None
DEFAULT_MATRIX_NPY = None
DEFAULT_ROTATE_XYZ = None         # Explicit rotation [rx, ry, rz] deg
DEFAULT_CURRENT_POSE = None       # [x, y, z, rx, ry, rz] for dry run
# -----------------------------------------------------------------------------------------


# Centralized primary CLI defaults using the constants above
PRIMARY_CLI_DEFAULTS = {
    'ip': DEFAULT_ROBOT_IP,
    'ip_help': 'robot IP address',
    'execute': DEFAULT_EXECUTE,
    'execute_help': 'actually send commands to the robot',
    'visualize': DEFAULT_VISUALIZE,
    'visualize_help': 'show realtime visualization (cv2)',
    'simple_loop': DEFAULT_SIMPLE_LOOP,
    'simple_loop_help': 'run continuous sensor->estimate loop',
    'use_sensor': DEFAULT_USE_SENSOR,
    'use_sensor_help': 'use the dmrobotics sensor input',
    'sensor_serial': DEFAULT_SENSOR_SERIAL,
    'sensor_serial_help': 'sensor device serial/id (when using --use-sensor or --simple-loop)',
    'save_frames': DEFAULT_SAVE_FRAMES,
    'save_frames_help': 'save visualization frames to outputs/realtime_rot_frames',
    'send_delta': DEFAULT_SEND_DELTA,
    'send_delta_help': 'If True send rotation deltas (omega) added to current rotvec; if False send smoothed raw rotations as absolute targets',
    'rotation_scale': DEFAULT_ROTATION_SCALE,
    'rotation_scale_help': 'scale factor applied to rotation when sending',
    'duration': DEFAULT_DURATION,
    'duration_help': 'duration (s) for loops; 0 means run forever',
    'interval': DEFAULT_INTERVAL,
    'interval_help': 'sleep interval (s) between sensor reads',
    # Advanced / Less frequently changed
    'matrix': DEFAULT_MATRIX,
    'matrix_help': 'comma/space-separated 9-number string for 3x3 matrix',
    'matrix_npy': DEFAULT_MATRIX_NPY,
    'matrix_npy_help': 'path to .npy file containing 3x3 matrix',
    'multiply': DEFAULT_MULTIPLY,
    'multiply_help': 'Legacy multiplication order: left or right',
    'current_pose': DEFAULT_CURRENT_POSE,
    'current_pose_help': 'current pose (x y z rx ry rz) for dry run',
    'ppmm': DEFAULT_PPMM,
    'shear_scale': DEFAULT_SHEAR_SCALE,
    'baseline_frames': DEFAULT_BASELINE_FRAMES,
    'depth_in_mm': DEFAULT_DEPTH_IN_MM,
    'contact_thresh_mm': DEFAULT_CONTACT_THRESH_MM,
    'angle_scale': DEFAULT_ANGLE_SCALE,
    'angle_scale_help': 'legacy shared angle scale used as fallback for all three rotation axes',
    'angle_scale_x': DEFAULT_ANGLE_SCALE_X,
    'angle_scale_x_help': 'scale factor applied to rotation around X',
    'angle_scale_y': DEFAULT_ANGLE_SCALE_Y,
    'angle_scale_y_help': 'scale factor applied to rotation around Y',
    'angle_scale_z': DEFAULT_ANGLE_SCALE_Z,
    'angle_scale_z_help': 'scale factor applied to rotation around Z',
    'frame': DEFAULT_FRAME,
    'frame_help': "Reference frame for rotation: 'global' (base) or 'local' (tool/end-effector).",
    'no_invert_rotation': DEFAULT_NO_INVERT_ROTATION,
    'no_invert_rotation_help': 'If set, send the same rotation direction as the estimator (default negates for opposite TCP motion).',
    'min_change_deg': DEFAULT_MIN_CHANGE_DEG,
    'min_change_deg_help': 'legacy shared minimum change in degrees used as fallback for all three rotation axes',
    'min_change_deg_x': DEFAULT_MIN_CHANGE_DEG_X,
    'min_change_deg_x_help': 'minimum change in degrees for rotation around X; below this threshold omega_x is forced to zero',
    'min_change_deg_y': DEFAULT_MIN_CHANGE_DEG_Y,
    'min_change_deg_y_help': 'minimum change in degrees for rotation around Y; below this threshold omega_y is forced to zero',
    'min_change_deg_z': DEFAULT_MIN_CHANGE_DEG_Z,
    'min_change_deg_z_help': 'minimum change in degrees for rotation around Z; below this threshold omega_z is forced to zero',
    'viz_axis_mapping': list(DEFAULT_VIZ_AXIS_MAPPING),
    'viz_axis_mapping_help': "visualizer axis remap for input xyz -> displayed XYZ; supports +/- sign, e.g. 'z x -y'",
    'min_translation_delta_m': DEFAULT_MIN_TRANSLATION_DELTA_M,
    'min_translation_delta_m_help': 'minimum norm of scaled translation increment (meters) per step; below this threshold translation is ignored',
    'min_translation_delta_m_x': DEFAULT_MIN_TRANSLATION_DELTA_M_X,
    'min_translation_delta_m_x_help': 'minimum scaled translation increment on X axis (meters) per step; below this threshold dx is ignored',
    'min_translation_delta_m_y': DEFAULT_MIN_TRANSLATION_DELTA_M_Y,
    'min_translation_delta_m_y_help': 'minimum scaled translation increment on Y axis (meters) per step; below this threshold dy is ignored',
    'min_translation_delta_m_z': DEFAULT_MIN_TRANSLATION_DELTA_M_Z,
    'min_translation_delta_m_z_help': 'minimum scaled translation increment on Z axis (meters) per step; below this threshold dz is ignored',
    'translation_scale': DEFAULT_TRANSLATION_SCALE,
    'translation_scale_help': 'shared scale factor applied to translation increment on all axes',
    'translation_scale_x': DEFAULT_TRANSLATION_SCALE_X,
    'translation_scale_x_help': 'scale factor applied to translation increment on X',
    'translation_scale_y': DEFAULT_TRANSLATION_SCALE_Y,
    'translation_scale_y_help': 'scale factor applied to translation increment on Y',
    'translation_scale_z': DEFAULT_TRANSLATION_SCALE_Z,
    'translation_scale_z_help': 'scale factor applied to translation increment on Z',
    'rotate_xyz': DEFAULT_ROTATE_XYZ,
    'rotate_xyz_help': "Apply explicit rotation (in degrees) around X, Y, Z axes of the chosen frame.",
    'tool_offset_z': DEFAULT_TOOL_OFFSET_Z,
    'tool_offset_z_help': "Distance from robot TCP (flange) to actual tool tip/pivot point in meters. Positive Z axis of tool.",
    'smooth_window': DEFAULT_SMOOTH_WINDOW,
    'smooth_window_help': "Number of frames for smoothing window (half size, total buffer is 2*N)",
    'use_depth_variation': DEFAULT_USE_DEPTH_VARIATION,
    'use_depth_variation_help': 'when set, use depth-temporal-variation based twist estimator (w from depth diff) instead of shear-only estimator',
    'contact_center_translation': DEFAULT_CONTACT_CENTER_TRANSLATION,
    'contact_center_translation_help': 'If set, fused XY translation comes from contact-mask centroid motion; disable joint tx,ty in shear/twist estimators.',
    'contact_center_translation_mode': DEFAULT_CONTACT_CENTER_TRANSLATION_MODE,
    'contact_center_translation_mode_help': "Centroid delta mode: 'consecutive' or 'window' (window uses --smooth-window).",
    'axis_order': DEFAULT_AXIS_ORDER,
    'axis_order_help': 'Three ints mapping robot X,Y,Z to sensor axes (indices 0,1,2). Example [1,2,0]',
    'axis_sign': DEFAULT_AXIS_SIGN,
    'axis_sign_help': 'Three ints (+1 or -1) applied to robot X,Y,Z after mapping to flip signs if needed',
    'translation_axis_order': DEFAULT_TRANSLATION_AXIS_ORDER,
    'translation_axis_order_help': 'Three ints mapping translation X,Y,Z to sensor axes (indices 0,1,2). Default identity [0,1,2]',
    'translation_axis_sign': DEFAULT_TRANSLATION_AXIS_SIGN,
    'translation_axis_sign_help': 'Three ints (+1 or -1) applied to translation X,Y,Z after translation-axis mapping',
}


def add_primary_cli_args(p: argparse.ArgumentParser) -> None:
    """Add all CLI options to the given ArgumentParser using defaults from PRIMARY_CLI_DEFAULTS."""
    defaults = globals().get('PRIMARY_CLI_DEFAULTS', {})

    def arg(name, **kwargs):
        key = name.replace('-', '_')
        if 'default' not in kwargs:
            kwargs['default'] = defaults.get(key)
        if 'help' not in kwargs and (key + '_help') in defaults:
            kwargs['help'] = defaults.get(key + '_help')
        p.add_argument('--' + name, **kwargs)

    arg('ip')
    arg('execute', action='store_true')
    arg('visualize', action='store_true')
    arg('simple-loop', action='store_true')
    arg('use-sensor', action='store_true')
    arg('sensor-serial', type=str)
    arg('save-frames', action='store_true')
    arg('send-delta', action='store_true')
    arg('rotation-scale', type=float)
    arg('duration', type=float)
    arg('interval', type=float)
    arg('matrix', type=str)
    arg('matrix-npy', type=str)
    arg('multiply', choices=['left', 'right'])
    arg('current-pose', nargs=6, type=float)
    arg('ppmm', type=float)
    arg('shear-scale', type=float)
    arg('baseline-frames', type=int)
    arg('depth-in-mm', action='store_true')
    arg('contact-thresh-mm', type=float)
    arg('angle-scale', type=float)
    arg('angle-scale-x', type=float)
    arg('angle-scale-y', type=float)
    arg('angle-scale-z', type=float)
    arg('frame', choices=['global', 'local'])
    arg('no-invert-rotation', action='store_true')
    arg('rotate-xyz', nargs=3, type=float, metavar=('RX', 'RY', 'RZ'))
    arg('tool-offset-z', type=float)
    arg('smooth-window', type=int)
    arg('use-depth-variation', action='store_true')
    arg('contact-center-translation', action='store_true')
    arg('contact-center-translation-mode', choices=['consecutive', 'window'])
    arg('axis-order', nargs=3, type=int, metavar=('A1', 'A2', 'A3'))
    arg('axis-sign', nargs=3, type=int, metavar=('S1', 'S2', 'S3'))
    arg('translation-axis-order', nargs=3, type=int, metavar=('TA1', 'TA2', 'TA3'))
    arg('translation-axis-sign', nargs=3, type=int, metavar=('TS1', 'TS2', 'TS3'))
    arg('min-change-deg', type=float)
    arg('min-change-deg-x', type=float)
    arg('min-change-deg-y', type=float)
    arg('min-change-deg-z', type=float)
    arg('viz-axis-mapping', nargs=3, type=str, metavar=('VX', 'VY', 'VZ'))
    arg('min-translation-delta-m', type=float)
    arg('min-translation-delta-m-x', type=float)
    arg('min-translation-delta-m-y', type=float)
    arg('min-translation-delta-m-z', type=float)
    arg('translation-scale', type=float)
    arg('translation-scale-x', type=float)
    arg('translation-scale-y', type=float)
    arg('translation-scale-z', type=float)


def simple_sensor_loop_cli(args, robot=None):
    """
    Run the continuous sensor loop. Refactored to use DaimonMonitor.
    """
    # Contact-mask centroid translation and the cumulative stdout line live only in the
    # legacy loop. Skip DaimonMonitor when enabled, otherwise the headless monitor runs
    # with almost no terminal output (only log.debug in the inner loop).
    _use_contact_center_tr = bool(
        getattr(args, 'contact_center_translation', DEFAULT_CONTACT_CENTER_TRANSLATION)
    )
    # 1. Try using the new Core logic
    # Note: if the user asked for visualization (--visualize), prefer the
    # legacy visual loop which contains cv2.imshow; the refactored
    # DaimonMonitor is headless.
    if (
        CORE_AVAILABLE
        and DaimonMonitor
        and not getattr(args, 'visualize', False)
        and not _use_contact_center_tr
    ):
        # Parse serial numbers 
        raw_serial = args.sensor_serial if hasattr(args, 'sensor_serial') and args.sensor_serial else None
        default_serial = globals().get('DEFAULT_SENSOR_SERIAL', None)
        
        # fallback to DEFAULT_SERIAL just in case old constant is around
        if not default_serial:
            default_serial = globals().get('DEFAULT_SERIAL', None)

        serial_str = raw_serial if raw_serial is not None else default_serial
        if not serial_str:
            print("Error: No sensor serial provided. Use --sensor-serial or set DEFAULT_SENSOR_SERIAL.")
            return

        serials = [x.strip() for x in serial_str.replace(',', ' ').split() if x.strip()]
        
        try:
            log.info("Initializing DaimonMonitor (Refactored Core)...")
            angle_scale_shared = float(getattr(args, 'angle_scale', DEFAULT_ANGLE_SCALE))
            angle_scale_vec = np.array([
                float(getattr(args, 'angle_scale_x', angle_scale_shared)),
                float(getattr(args, 'angle_scale_y', angle_scale_shared)),
                float(getattr(args, 'angle_scale_z', angle_scale_shared)),
            ], dtype=float)
            monitor = DaimonMonitor(
                sensor_serials=serials,
                robot=robot,
                ppmm=getattr(args, 'ppmm', DEFAULT_PPMM),
                shear_scale=getattr(args, 'shear_scale', DEFAULT_SHEAR_SCALE),
                angle_scale=angle_scale_vec,
                smooth_window=getattr(args, 'smooth_window', DEFAULT_SMOOTH_WINDOW),
                baseline_frames=getattr(args, 'baseline_frames', DEFAULT_BASELINE_FRAMES),
                depth_in_mm=getattr(args, 'depth_in_mm', DEFAULT_DEPTH_IN_MM),
                use_divfree_only=getattr(args, 'use_divfree_only', DEFAULT_USE_DIVFREE_ONLY),
                interval=getattr(args, 'interval', DEFAULT_INTERVAL),
                execute=getattr(args, 'execute', False),
                tool_offset_z=getattr(args, 'tool_offset_z', DEFAULT_TOOL_OFFSET_Z),
                frame=getattr(args, 'frame', DEFAULT_FRAME),
                invert_rotation=not getattr(args, 'no_invert_rotation', DEFAULT_NO_INVERT_ROTATION),
                send_swap_xz=True,
                contact_thresh_mm=float(getattr(args, 'contact_thresh_mm', DEFAULT_CONTACT_THRESH_MM)),
                delta_scale=float(getattr(args, 'rotation_scale', DEFAULT_ROTATION_SCALE)),
                send_delta=bool(getattr(args, 'send_delta', DEFAULT_SEND_DELTA)),
                axis_order=list(getattr(args, 'axis_order', DEFAULT_AXIS_ORDER)),
                axis_sign=list(getattr(args, 'axis_sign', DEFAULT_AXIS_SIGN)),
                max_step_deg=float(getattr(args, 'max_step_deg', DEFAULT_MAX_STEP_DEG)),
                include_translation=(
                    DEFAULT_INCLUDE_TRANSLATION
                    and not bool(getattr(args, 'contact_center_translation', DEFAULT_CONTACT_CENTER_TRANSLATION))
                ),
            )
            # Add logging before loop
            log.info("DaimonMonitor initialized. Starting loop...")
            monitor.loop(duration=getattr(args, 'duration', 0.0))
            monitor.shutdown()
            log.info("DaimonMonitor shutdown complete.")
            return
        except Exception as e:
            log.error(f"DaimonMonitor failed: {e}")
            log.info("Falling back to legacy monolithic loop...")
            # If monitor fails (e.g. import error inside), fall through to legacy code below

    # =============================================================================================
    # LEGACY / FALLBACK IMPLEMENTATION
    # =============================================================================================
    
    # lazily import estimator/sensor pieces
    try:
        from dmrobotics import Sensor as _Sensor
    except Exception:
        _Sensor = None
    if _Sensor is None:
        print('Sensor not available. Install `dmrobotics` to run the realtime loop.')
        return

    raw_serial = args.sensor_serial if hasattr(args, 'sensor_serial') and args.sensor_serial else None
    default_serial = globals().get('DEFAULT_SENSOR_SERIAL', None)
    
    # If using correct variable name from update: DEFAULT_SENSOR_SERIAL
    # fallback to DEFAULT_SERIAL just in case old constant is around
    if not default_serial:
        default_serial = globals().get('DEFAULT_SERIAL', None)

    serial_str = raw_serial if raw_serial is not None else default_serial
    if not serial_str:
        raise SystemExit('Sensor requires a serial/dev_id; provide --sensor-serial or set DEFAULT_SENSOR_SERIAL in the file')
    
    # Split by space or comma to get list of serials
    serials = [x.strip() for x in serial_str.replace(',', ' ').split() if x.strip()]
    
    # helper: guarded create/connect with timeout to avoid blocking on resetting devices
    sensors = []
    log.info(f"Initializing {len(serials)} sensors: {serials}")
    for ser in serials:
        try:
            def _create_and_connect(sid):
                obj = _Sensor(sid)
                if hasattr(obj, 'connect'):
                    obj.connect()
                return obj

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_create_and_connect, ser)
                try:
                    s_obj = fut.result(timeout=3.0)
                    sensors.append(s_obj)
                except concurrent.futures.TimeoutError:
                    log.warning(f"Timeout initializing sensor {ser}; skipping for now.")
                except Exception as e:
                    log.warning(f"Failed to initialize sensor {ser}: {e}")
        except Exception as e:
            log.warning(f"Failed to initialize sensor {ser}: {e}")

    if not sensors:
         raise SystemExit("No sensors could be initialized.")

    use_contact_center_tr = bool(
        getattr(args, 'contact_center_translation', DEFAULT_CONTACT_CENTER_TRANSLATION)
    )
    contact_center_state = None
    if use_contact_center_tr:
        if ContactCenterTranslationState is not None:
            _cc_mode = getattr(args, 'contact_center_translation_mode', DEFAULT_CONTACT_CENTER_TRANSLATION_MODE)
            contact_center_state = ContactCenterTranslationState(
                len(sensors),
                mode=_cc_mode,
                smooth_window=int(getattr(args, 'smooth_window', DEFAULT_SMOOTH_WINDOW)),
            )
            log.info(
                'Contact-center translation: mode=%s smooth_window=%s',
                _cc_mode,
                int(getattr(args, 'smooth_window', DEFAULT_SMOOTH_WINDOW)),
            )
        else:
            log.warning(
                'contact_center_translation requested but ContactCenterTranslationState import failed; '
                'using estimator translation'
            )
            use_contact_center_tr = False

    # optional arrow helper from dmrobotics
    try:
        from dmrobotics import put_arrows_on_image as _put_arrows_on_image
    except Exception:
        _put_arrows_on_image = None

    # optional pose visualizer (module-level PoseVisualizer imported earlier)
    pv = None
    if getattr(args, 'visualize', False) and PoseVisualizer is not None:
        try:
            min_change_shared = float(getattr(args, 'min_change_deg', DEFAULT_MIN_CHANGE_DEG))
            pv = PoseVisualizer(
                min_change_deg=[
                    float(getattr(args, 'min_change_deg_x', min_change_shared)),
                    float(getattr(args, 'min_change_deg_y', min_change_shared)),
                    float(getattr(args, 'min_change_deg_z', min_change_shared)),
                ],
                axis_mapping=tuple(getattr(args, 'viz_axis_mapping', DEFAULT_VIZ_AXIS_MAPPING)),
            )
        except Exception:
            pv = None
    # Robot worker (background thread) to avoid blocking main GUI loop
    worker = None
    send_fn = None
    if getattr(args, 'execute', False) and robot is not None:
        try:
            class RobotWorker(threading.Thread):
                """Background worker that manages robot connection, fetches current TCP
                pose, computes actions and sends them. Receives estimation requests
                from the main loop (overwrite-latest semantics).
                """
                def __init__(self, robot):
                    super().__init__(daemon=True)
                    self.robot = robot
                    self._stop_evt = threading.Event()
                    # remember initial connected state to avoid double connect/disconnect
                    self._initially_connected = bool(getattr(robot, 'is_connected', False))
                    self.connected = self._initially_connected
                    # latest request to process (overwrite semantics)
                    self._latest_req = None
                    self._evt = threading.Event()

                def send(self, req: dict):
                    """Queue the latest estimation request (overwrite previous) and notify worker.

                    Expected req keys:
                      - 'R': rotation matrix (3x3) OR 'omega_rad': 3-vector; prefer R when present
                      - 'send_delta': bool
                      - 'delta_scale': float
                      - 'multiply': 'left'|'right'
                      - 'frame': 'global'|'local'
                    """
                    self._latest_req = req
                    self._evt.set()

                def stop(self):
                    self._stop_evt.set()
                    self._evt.set()

                def run(self):
                    try:
                        if not self._initially_connected:
                            try:
                                self.robot.connect()
                                self.connected = True
                                log.info('RobotWorker connected')
                            except Exception as e:
                                log.warning('RobotWorker initial connect failed: %s', e)
                    except Exception as e:
                        log.warning('RobotWorker setup failed: %s', e)
                    while not self._stop_evt.is_set():
                        try:
                            got = self._evt.wait(timeout=0.2)
                        except Exception:
                            got = False
                        if not got:
                            continue
                        self._evt.clear()
                        if self._stop_evt.is_set():
                            break
                        req = self._latest_req
                        if req is None:
                            continue
                        # ensure connected
                        if not self.connected:
                            try:
                                self.robot.connect()
                                self.connected = True
                                log.info('RobotWorker connected')
                            except Exception as e:
                                log.warning('RobotWorker connect failed: %s', e)
                                # skip this request; wait for next
                                continue
                        # fetch current pose and compute action here to avoid blocking main thread
                        try:
                            t0 = time.time()
                            cur = self.robot.r_inter.getActualTCPPose()
                            t1 = time.time()
                        except Exception as e:
                            log.warning('RobotWorker getActualTCPPose failed: %s', e)
                            continue
                        try:
                            # Unified logic: Always use compute_action_from_pose.
                            # If send_delta is True, we interpret the input as a "small delta rotation"
                            # to be applied to the current pose (in the chosen frame).
                            if req.get('send_delta', False):
                                omega_rad = req.get('omega_rad')
                                if omega_rad is None:
                                    # if only R is provided, try to extract omega, or just treat R as the delta
                                    # For safety, if omage_rad is missing in delta mode, we default to 0
                                    omega_rad = np.zeros(3, dtype=float)
                                
                                # Apply scaling to the delta vector
                                scale = float(req.get('delta_scale', DEFAULT_ROTATION_SCALE))
                                rvec_delta = np.asarray(omega_rad, dtype=float) * scale
                                
                                # Convert the scaled delta vector to a Rotation Matrix
                                R_use = rotvec_to_matrix(rvec_delta)
                            else:
                                # Absolute target mode (or raw R application)
                                R_use = req.get('R')
                                if R_use is None and 'omega_rad' in req:
                                    try:
                                        R_use = euler_angles_to_rotation_matrix(np.asarray(req['omega_rad'], dtype=float))
                                    except Exception:
                                        R_use = np.eye(3)

                            # Now apply this rotation (whether delta or absolute) using the correct frame/offset logic
                            action = compute_action_from_pose(
                                cur[:3], 
                                cur[3:6], 
                                R_use, 
                                multiply=req.get('multiply', 'left'), 
                                frame=req.get('frame', 'global'),
                                tool_offset_z=float(req.get('tool_offset_z', 0.0))
                            )
                            
                            log.debug('RobotWorker computed action: %s', action)
                            try:
                                self.robot.send_action(action)
                                log.debug('RobotWorker send_action returned')
                            except Exception as e:
                                log.warning('RobotWorker send_action failed: %s', e)
                        except Exception as e:
                            log.warning('RobotWorker failed to compute/send action: %s', e)
                    # cleanup: only disconnect if we connected here
                    try:
                        if not self._initially_connected and self.connected:
                            self.robot.disconnect()
                    except Exception:
                        pass

            worker = RobotWorker(robot)
            worker.start()
            send_fn = lambda a: worker.send(a)
        except Exception:
            worker = None
            send_fn = (lambda a: None)
    else:
        send_fn = (lambda a: None)
    # prepare frame saving if requested
    frame_idx = 0
    save_dir = None
    if getattr(args, 'save_frames', False):
        try:
            import os
            save_dir = os.path.join(os.getcwd(), 'outputs', 'realtime_rot_frames')
            os.makedirs(save_dir, exist_ok=True)
            log.info('Saving visualization frames to %s', save_dir)
        except Exception:
            save_dir = None

    try:
        baselines = [None] * len(sensors)
        if getattr(args, 'baseline_frames', 0) and args.baseline_frames > 0:
            log.info(f"Acquiring baseline frames for {len(sensors)} sensors...")
            for i, sens in enumerate(sensors):
                try:
                    depths = [np.array(sens.getDepth(), dtype=np.float32) for _ in range(args.baseline_frames)]
                    baselines[i] = np.mean(np.stack(depths, axis=0), axis=0) * (1000.0 if not getattr(args, 'depth_in_mm', False) else 1.0)
                except Exception as e:
                    log.warning(f"Baseline acquisition failed for sensor {i}: {e}")

        # Counter for warmup frames
        valid_frames_count = 0
        WARMUP_FRAMES = DEFAULT_WARMUP_FRAMES
        # Buffer for smoothing
        smooth_window = getattr(args, 'smooth_window', DEFAULT_SMOOTH_WINDOW)
        # buffer size needs to be 2 * smooth_window to compare "front 10" and "back 10"
        # actually, if we want to "compare average of front 10 and back 10", we need at least 20 frames if they are disjoint,
        # or just a sliding window.
        # User said: "contrast average value of front 10 frames and back 10 frames"
        # Let's interpret this as: maintain a buffer of history. 
        # But for real-time control, usually we just want to smooth the CURRENT input.
        # If the user wants to COMPARE them and SEND THE CHANGE, maybe they mean:
        # buffer = [oldest ... newest]
        # avg_old = mean(buffer[:10])
        # avg_new = mean(buffer[-10:])
        # diff = avg_new - avg_old  <-- this is the "change" or "trend"
        # But standard control usually applies the CURRENT estimated rotation.
        # If the estimator outputs "current rotation from sensor zero", then smoothing the absolute value is better.
        # If the estimator outputs "delta", then averaging deltas is smoothing velocity.

        # Let's implement the buffer.
        rot_buffer = [] 
        buffer_size = 2 * smooth_window
        cumulative_angle_deg = np.zeros(3, dtype=float)
        cumulative_translation_m = np.zeros(3, dtype=float)

        # storage for previous depth frames per sensor when using depth-variation estimator
        prev_depths = [None] * len(sensors)
        # ring buffers of recent depth frames for windowed averaging (per sensor)
        depth_buffers = [deque(maxlen=buffer_size) for _ in sensors]

        end = time.time() + args.duration if getattr(args, 'duration', 0.0) and args.duration > 0 else None
        loop_started_wall = time.time()
        last_no_contact_notice = 0.0
        log.info(
            'Sensor loop started (%d sensor(s)). Status line prints once there is contact on at least one sensor.',
            len(sensors),
        )
        try:
            print(
                '[TacSE3] Loop running; waiting for contact (no output until a non-empty mask)…',
                flush=True,
            )
        except Exception:
            pass
        while True:
            if end is not None and time.time() >= end:
                break
            try:
                est_include_translation = DEFAULT_INCLUDE_TRANSLATION and not use_contact_center_tr
                sensor_masks_for_center = [None] * len(sensors)
                # ---------------------
                # Read multiple sensors
                # ---------------------
                all_images = []
                all_depth_m = []
                all_shear_m = []
                all_masks = []
                all_rotations_rad = []
                all_translation_delta_m = []
                
                valid_sensor_data = False
                
                # Iterate over ALL sensors
                for i, s in enumerate(sensors):
                    try:
                        # read sensor frames (raw image optional)
                        try:
                            img = s.getRawImage()
                        except Exception:
                            img = None
                        # guarded depth/shear read to avoid blocking if sensor is resetting
                        try:
                            def _read(sobj):
                                return sobj.getDepth(), sobj.getShear()
                            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as rex:
                                rfut = rex.submit(_read, s)
                                depth, shear = rfut.result(timeout=max(0.1, args.interval * 2.0))
                        except Exception:
                            depth = None
                            shear = None
                        
                        # 新增：获取传感器旋转角（如果有）
                        sensor_rot = None
                        if hasattr(s, 'getRotation'):
                            sensor_rot = s.getRotation()
                        
                        all_images.append(img)
                        
                        # Build contact
                        try:
                            baseline = baselines[i] if i < len(baselines) else None
                            # shear_scale is applied inside ``estimate_rotation_from_shear`` only (not here), matching legacy loop.
                            depth_m, shear_m, C_mask = build_contact_from_daimon(
                                depth,
                                shear,
                                args.ppmm,
                                depth_in_mm=getattr(args, 'depth_in_mm', DEFAULT_DEPTH_IN_MM),
                                shear_scale=1.0,
                                contact_thresh_mm=float(getattr(args, 'contact_thresh_mm', DEFAULT_CONTACT_THRESH_MM)),
                                baseline_depth_mm=baseline,
                                press_increases_depth=True,
                            )
                        except Exception as e:
                            log.warning(f'build_contact failed for sensor {i}: {e}')
                            all_depth_m.append(None)
                            all_shear_m.append(None)
                            all_masks.append(None)
                            sensor_masks_for_center[i] = None
                            continue
                        
                        if not np.any(C_mask):
                            # No contact for this sensor
                            sensor_masks_for_center[i] = C_mask
                            all_depth_m.append(depth_m)
                            all_shear_m.append(shear_m)
                            all_masks.append(C_mask)
                            continue
                            
                        # If user requested the depth-variation estimator, try that first.
                        omega_rad_one = None
                        t_vec_one = None
                        t_is_velocity = False
                        if getattr(args, 'use_depth_variation', False):
                            try:
                                # Need previous depth for this sensor
                                prev = prev_depths[i]
                                if prev is not None:
                                    # Build point list from contact mask: take pixel coordinates (x,y) in metric using pixel_xy_grid
                                    # Reuse pixel grid helper from estimate_rot_from_shear module if available
                                    try:
                                        Xg, Yg = pixel_xy_grid(depth_m.shape, args.ppmm)
                                    except Exception:
                                        # fallback: create approximate metric grid centered
                                        H, W = depth_m.shape
                                        xv = np.arange(W) - (W / 2.0) + 0.5
                                        yv = np.arange(H) - (H / 2.0) + 0.5
                                        xv, yv = np.meshgrid(xv, yv)
                                        scale = (1.0 / args.ppmm) * 1e-3
                                        Xg = xv * scale
                                        Yg = yv * scale

                                    yy, xx = np.nonzero(C_mask)
                                    if yy.size > 5:
                                        # sample up to 500 points evenly to keep LS small
                                        idxs = np.arange(yy.size)
                                        if yy.size > 500:
                                            step = max(1, yy.size // 500)
                                            idxs = idxs[::step]
                                        sel_y = yy[idxs]
                                        sel_x = xx[idxs]
                                        pts_xy = np.stack([Xg[sel_y, sel_x], Yg[sel_y, sel_x]], axis=1)
                                        # u,v from shear_m at those pixels
                                        us = shear_m[sel_y, sel_x, 0]
                                        vs = shear_m[sel_y, sel_x, 1]
                                        # Determine depth_now / depth_prev using windowed averaging if buffer full
                                        dbuf = depth_buffers[i]
                                        if len(dbuf) >= buffer_size:
                                            # average first N and last N frames
                                            first_stack = np.stack(list(dbuf)[:smooth_window], axis=0)
                                            last_stack = np.stack(list(dbuf)[-smooth_window:], axis=0)
                                            avg_first = np.mean(first_stack, axis=0)
                                            avg_last = np.mean(last_stack, axis=0)
                                            depth_now = avg_last[sel_y, sel_x]
                                            depth_prev_sel = avg_first[sel_y, sel_x]
                                            # dt across the window: smooth_window * interval
                                            dt_est = max(1e-6, float(getattr(args, 'interval', DEFAULT_INTERVAL)) * float(smooth_window))
                                        else:
                                            # fallback to single-frame previous behavior
                                            depth_now = depth_m[sel_y, sel_x]
                                            depth_prev_sel = prev[sel_y, sel_x]
                                            dt_est = max(1e-6, float(getattr(args, 'interval', DEFAULT_INTERVAL)))

                                        est = estimate_twist_from_depth_variation(pts_xy, us, vs, depth_now, depth_prev_sel, dt_est)
                                        omega_rad_one = np.array(est['omega'], dtype=float)
                                        if not use_contact_center_tr:
                                            try:
                                                t_vec_one = np.array(est.get('t', np.zeros(3, dtype=float)), dtype=float).reshape(3,)
                                                t_is_velocity = True
                                            except Exception:
                                                t_vec_one = np.zeros(3, dtype=float)
                                                t_is_velocity = True
                                        # record residuals if needed
                                        # log.debug('Depth-var estimator resid=%s', est['residuals'])
                                # update prev and depth buffer regardless
                                prev_depths[i] = depth_m.copy()
                                try:
                                    depth_buffers[i].append(depth_m.copy())
                                except Exception:
                                    pass
                            except Exception as e:
                                log.warning(f'depth-variation estimator failed for sensor {i}: {e}')
                                prev_depths[i] = depth_m.copy()

                        # Fallback to original shear-based estimator if we didn't compute omega
                        if omega_rad_one is None:
                            res = estimate_rotation_from_shear(
                                depth_m,
                                shear_m,
                                ppmm=args.ppmm,
                                contact_mask=C_mask,
                                shear_scale=float(getattr(args, 'shear_scale', DEFAULT_SHEAR_SCALE)),
                                include_translation=est_include_translation,
                                use_divfree_only=getattr(args, 'use_divfree_only', DEFAULT_USE_DIVFREE_ONLY),
                            )
                            omega_rad_one = np.array(res['omega_rad'], dtype=float)
                            try:
                                if isinstance(res, dict) and ('translation_m' in res) and (res['translation_m'] is not None):
                                    t_xy = np.array(res['translation_m'], dtype=float).ravel()
                                    t_xy = np.pad(t_xy, (0, max(0, 2 - t_xy.size)))
                                    t_vec_one = np.array([t_xy[0], t_xy[1], 0.0], dtype=float)
                                    t_is_velocity = False
                            except Exception:
                                t_vec_one = None
                        
                        # Use hardware rotation if available
                        if sensor_rot is not None:
                             try:
                                sensor_rot_arr = np.array(sensor_rot, dtype=float)
                                if sensor_rot_arr.shape == (3,):
                                    # log.info(f"Using direct sensor rotation sensor {i}: {sensor_rot_arr}")
                                    omega_rad_one = sensor_rot_arr
                             except Exception:
                                pass

                        all_rotations_rad.append(omega_rad_one)
                        if t_vec_one is not None and not use_contact_center_tr:
                            t_vec_one = np.asarray(t_vec_one, dtype=float).ravel()
                            t_vec_one = np.pad(t_vec_one, (0, max(0, 3 - t_vec_one.size)))[:3]
                            if t_is_velocity:
                                dt_tmp = max(0.0, float(getattr(args, 'interval', DEFAULT_INTERVAL)))
                                all_translation_delta_m.append(t_vec_one * dt_tmp)
                            else:
                                all_translation_delta_m.append(t_vec_one)
                        
                        all_depth_m.append(depth_m)
                        all_shear_m.append(shear_m)
                        all_masks.append(C_mask)
                        sensor_masks_for_center[i] = C_mask
                        
                        valid_sensor_data = True
                        
                    except Exception as e:
                        log.warning(f'sensor {i} read/process failed: {e}')
                        all_images.append(None)
                        all_depth_m.append(None)
                        all_shear_m.append(None)
                        all_masks.append(None)
                        try:
                            sensor_masks_for_center[i] = None
                        except Exception:
                            pass
                        continue
                
                if not valid_sensor_data:
                    # Clear buffer if ALL sensors lost contact
                    if len(rot_buffer) > 0:
                         rot_buffer = []
                    # No frame reached the estimator: no \r status line this cycle — notify periodically.
                    _now = time.time()
                    if _now - last_no_contact_notice >= 2.0:
                        last_no_contact_notice = _now
                        log.info(
                            'Waiting for contact or valid read (%.0fs elapsed; all masks empty or read failed).',
                            _now - loop_started_wall,
                        )
                        try:
                            print(
                                f'\n[TacSE3] No contact / no valid frame yet '
                                f'({_now - loop_started_wall:.1f}s). Press surface or check sensor/USB.',
                                flush=True,
                            )
                        except Exception:
                            pass
                    # Keep cumulative angle until user explicitly resets it with space.
                    time.sleep(max(0.01, args.interval))
                    continue
                
                # Averaging / Fusion Strategy
                if len(all_rotations_rad) == 2:
                    # 针对两个传感器的镜像融合策略
                    # "两个是先减再平均，有一个是相加再平均"
                    # 假设 Sensor 1 (all_rotations_rad[1]) 相对于 Sensor 0 (all_rotations_rad[0]) 是镜像的
                    # 如果某轴是镜像的（反向），则 Sensor 1 读数为 -Ref。 Ref = (S0 - S1) / 2
                    # 如果某轴是同向的，则 Sensor 1 读数为 +Ref。 Ref = (S0 + S1) / 2
                    
                    r0 = all_rotations_rad[0]
                    r1 = all_rotations_rad[1]
                    
                    # 融合系数：1.0 表示相加(同向)，-1.0 表示相减(反向)
                    # 默认假设：Y轴同向(1.0)，X和Z轴反向(-1.0)
                    # 您可以根据实际情况修改这个数组: [Rx_sign, Ry_sign, Rz_sign]
                    fusion_signs = np.array([1.0, -1.0, -1.0])
                    
                    omega_rad = (r0 + r1 * fusion_signs) / 2.0
                    
                    # log.debug(f"Fusion: S0={r0}, S1={r1}, Signs={fusion_signs}, Res={omega_rad}")
                    
                elif len(all_rotations_rad) > 0:
                    # 默认平均策略
                    omega_rad = np.mean(np.stack(all_rotations_rad, axis=0), axis=0)
                else:
                    omega_rad = np.zeros(3)

                if use_contact_center_tr and contact_center_state is not None:
                    try:
                        t_vec_delta_m = np.asarray(
                            contact_center_state.update(
                                sensor_masks_for_center,
                                float(getattr(args, 'ppmm', DEFAULT_PPMM)),
                            ),
                            dtype=float,
                        ).reshape(3,)
                    except Exception as e:
                        log.warning('contact-center translation update failed: %s', e)
                        t_vec_delta_m = np.zeros(3, dtype=float)
                elif len(all_translation_delta_m) == 2:
                    t0 = np.asarray(all_translation_delta_m[0], dtype=float)
                    t1 = np.asarray(all_translation_delta_m[1], dtype=float)
                    # Translation fusion for two sensors:
                    # X/Z use average; Y uses subtract-then-average.
                    t_vec_delta_m = (t0 + t1) / 2.0
                    t_vec_delta_m[1] = (t0[1] - t1[1]) / 2.0
                elif len(all_translation_delta_m) > 0:
                    t_vec_delta_m = np.mean(np.stack(all_translation_delta_m, axis=0), axis=0)
                else:
                    t_vec_delta_m = np.zeros(3, dtype=float)

            except Exception as e:
                log.warning('main loop exception: %s', e)
                time.sleep(max(0.01, args.interval))
                continue

            # --- Axis Remapping to robot frame ---
            try:
                order = getattr(args, 'axis_order', DEFAULT_AXIS_ORDER)
                sign = getattr(args, 'axis_sign', DEFAULT_AXIS_SIGN)
                # normalize to lists of length 3
                if order is None:
                    order = DEFAULT_AXIS_ORDER
                if sign is None:
                    sign = DEFAULT_AXIS_SIGN
                order = [int(x) for x in order]
                sign = [int(x) for x in sign]
                omega_raw = np.asarray(omega_rad, dtype=float)
                omega_remap = np.array([omega_raw[i] for i in order], dtype=float) * np.asarray(sign, dtype=float)
                omega_rad = omega_remap
                t_order = getattr(args, 'translation_axis_order', DEFAULT_TRANSLATION_AXIS_ORDER)
                t_sign = getattr(args, 'translation_axis_sign', DEFAULT_TRANSLATION_AXIS_SIGN)
                if t_order is None:
                    t_order = DEFAULT_TRANSLATION_AXIS_ORDER
                if t_sign is None:
                    t_sign = DEFAULT_TRANSLATION_AXIS_SIGN
                t_order = [int(x) for x in t_order]
                t_sign = [int(x) for x in t_sign]
                t_raw = np.asarray(t_vec_delta_m, dtype=float)
                t_remap = np.array([t_raw[i] for i in t_order], dtype=float) * np.asarray(t_sign, dtype=float)
                t_vec_delta_m = t_remap
            except Exception:
                # fallback: keep original
                pass

            # Apply per-axis angle scale in robot-axis order (after remap).
            angle_scale_shared = float(getattr(args, 'angle_scale', DEFAULT_ANGLE_SCALE))
            angle_scale_xyz = np.array([
                float(getattr(args, 'angle_scale_x', angle_scale_shared)),
                float(getattr(args, 'angle_scale_y', angle_scale_shared)),
                float(getattr(args, 'angle_scale_z', angle_scale_shared)),
            ], dtype=float)
            omega_rad = np.asarray(omega_rad, dtype=float) * angle_scale_xyz

            # --- Smoothing Logic ---
            rot_buffer.append(omega_rad)
            if len(rot_buffer) > buffer_size:
                rot_buffer.pop(0)
            
            delta_omega = np.zeros(3, dtype=float)
            # Only proceed if we have enough data? Or use partial buffer?
            # User requirement: "compare average of front 10 and back 10"
            if len(rot_buffer) >= buffer_size:
                # split buffer
                first_half = rot_buffer[:smooth_window]
                last_half = rot_buffer[-smooth_window:]
                
                avg_first = np.mean(first_half, axis=0)
                avg_last = np.mean(last_half, axis=0)
                
                # "Send change to robot arm" -> implies the result we want to apply is related to this difference?
                # If we are sending absolute rotation, maybe they want the RESULT of this comparison.
                # If `omega_rad` from `estimate_rotation_from_shear` is used as a "velocity" or "incremental rotation",
                # then sending the change (diff) makes sense.
                
                # Let's assume we use this "diff" as the `omega_rad` to be processed.
                # ONLY IF this specific logic is desired. 
                # But the user put this request broadly.
                
                # Let's assume the user wants to use `avg_last` as the current estimate (smoothed)
                # AND they want to check if it changed enough? 
                
                # "compare ... send change" might just mean "send the difference".
                # Let's use `diff` as the `omega_rad` to be sent.
                
                # Wait, if I replace `omega_rad` with `diff`, I am changing the semantics of the variable.
                # `omega_rad` was absolute rotation of sensor.
                # `diff` is change in rotation of sensor.
                
                # If `send_delta` is True, then we want the delta.
                # If `send_delta` is False, we are sending ABSOLUTE target.
                # The user request seems to imply logic for `send_delta=True`.
                
                # So I should compute `diff = avg_last - avg_first` and use `diff` as the `omega_rad` to send.
                # AND likely remove the `delta_scale` or set it to 1.0 (or keep it as a gain).
                
                omega_rad_smoothed = avg_last

                # The "change" the user speaks of is the drift from the "old" window to "new" window.
                delta_omega = avg_last - avg_first

                # Choose whether to send the delta (existing behavior) or the smoothed raw rotation.
                if getattr(args, 'send_delta', DEFAULT_SEND_DELTA):
                    omega_rad = delta_omega
                else:
                    omega_rad = omega_rad_smoothed
            else:
                # Not enough data yet
                if len(rot_buffer) > 0:
                    # fallback to just current or mean of what we have
                    omega_rad = np.mean(rot_buffer, axis=0)
                else:
                    omega_rad = np.zeros(3)

            # --- Safety: Clamping large jumps ---
            # Use the same processed omega for visualization and accumulation so
            # the cumulative window does not visually lag behind the current one.
            omega_rad_vis = np.array(omega_rad, dtype=float)

            # Zero-out small motion using per-axis change thresholds.
            try:
                change_source = delta_omega if len(rot_buffer) >= buffer_size else omega_rad
                change_deg_xyz = np.abs(np.degrees(np.asarray(change_source, dtype=float)))
                min_change_shared = float(getattr(args, 'min_change_deg', DEFAULT_MIN_CHANGE_DEG))
                min_change_xyz = np.array([
                    float(getattr(args, 'min_change_deg_x', min_change_shared)),
                    float(getattr(args, 'min_change_deg_y', min_change_shared)),
                    float(getattr(args, 'min_change_deg_z', min_change_shared)),
                ], dtype=float)
                zero_mask = change_deg_xyz < min_change_xyz
                if np.any(zero_mask):
                    omega_rad = np.asarray(omega_rad, dtype=float)
                    omega_rad[zero_mask] = 0.0
                    log.debug(
                        "Zeroing omega axes: change_deg=%s, min_change_deg_xyz=%s, zero_mask=%s",
                        np.array2string(change_deg_xyz, precision=3),
                        np.array2string(min_change_xyz, precision=3),
                        zero_mask.tolist(),
                    )
            except Exception:
                pass

            omega_norm = np.linalg.norm(omega_rad)
            max_rad = np.radians(DEFAULT_MAX_STEP_DEG)
            if omega_norm > max_rad:
                scale_safe = max_rad / omega_norm
                omega_rad = omega_rad * scale_safe
                # log.warning(f"Clamped rotation step: {np.degrees(omega_norm):.1f}deg -> {DEFAULT_MAX_STEP_DEG}deg")

            # Opposite direction vs estimator (typical for tool-frame mapping); disable with --no-invert-rotation
            if not getattr(args, 'no_invert_rotation', DEFAULT_NO_INVERT_ROTATION):
                omega_rad = -np.asarray(omega_rad, dtype=float)

            # Robot-only remap: swap X/Z before sending action to robot.
            # Keep visualization / logs on omega_rad (pre-send mapping) for interpretability.
            omega_send_rad = np.asarray(omega_rad, dtype=float).copy()
            omega_send_rad = omega_send_rad[[2, 1, 0]]

            # Human-facing cumulative print keeps the non-send-swapped convention.
            omega_deg = np.degrees(omega_rad)
            cumulative_angle_deg = cumulative_angle_deg + omega_deg
            delta_translation_m = np.asarray(t_vec_delta_m, dtype=float)
            # Apply per-axis translation scale in robot-axis order (after remap).
            translation_scale_shared = float(getattr(args, 'translation_scale', DEFAULT_TRANSLATION_SCALE))
            translation_scale_xyz = np.array([
                float(getattr(args, 'translation_scale_x', translation_scale_shared)),
                float(getattr(args, 'translation_scale_y', translation_scale_shared)),
                float(getattr(args, 'translation_scale_z', translation_scale_shared)),
            ], dtype=float)
            delta_translation_m = np.asarray(delta_translation_m, dtype=float) * translation_scale_xyz
            try:
                min_translation_shared = float(
                    getattr(args, 'min_translation_delta_m', DEFAULT_MIN_TRANSLATION_DELTA_M)
                )
            except Exception:
                min_translation_shared = float(DEFAULT_MIN_TRANSLATION_DELTA_M)
            try:
                min_translation_xyz = np.array([
                    float(getattr(args, 'min_translation_delta_m_x', min_translation_shared)),
                    float(getattr(args, 'min_translation_delta_m_y', min_translation_shared)),
                    float(getattr(args, 'min_translation_delta_m_z', min_translation_shared)),
                ], dtype=float)
            except Exception:
                min_translation_xyz = np.array([min_translation_shared] * 3, dtype=float)
            trans_abs = np.abs(np.asarray(delta_translation_m, dtype=float))
            zero_t_mask = trans_abs < min_translation_xyz
            if np.any(zero_t_mask):
                delta_translation_m = np.asarray(delta_translation_m, dtype=float)
                delta_translation_m[zero_t_mask] = 0.0
            cumulative_translation_m = cumulative_translation_m + delta_translation_m
            try:
                print(
                    '\rCumulative rotation (deg) [rx, ry, rz]: '
                    f'{np.round(cumulative_angle_deg, 4)}'
                    ' | Cumulative translation (m) [x, y, z]: '
                    f'{np.round(cumulative_translation_m, 6)}',
                    end='',
                    flush=True,
                )
            except Exception:
                pass
            # Build robot command from send-mapped rotation (X/Z swapped).
            # In delta mode the worker applies delta_scale; in absolute mode we apply scale here.
            if getattr(args, 'send_delta', False):
                R = euler_angles_to_rotation_matrix(omega_send_rad)
            else:
                scale_val = float(getattr(args, 'rotation_scale', DEFAULT_ROTATION_SCALE))
                try:
                    rvec_scaled = np.asarray(omega_send_rad, dtype=float) * scale_val
                    R = rotvec_to_matrix(rvec_scaled)
                except Exception:
                    R = euler_angles_to_rotation_matrix(omega_send_rad)
            
            if getattr(args, 'execute', False) and robot is not None:
                try:
                    # Send a lightweight estimation request to the worker. The worker will
                    # fetch the latest robot pose, compute the full action, and send it.
                    rotation_scale_val = float(getattr(args, 'rotation_scale', DEFAULT_ROTATION_SCALE))
                    req = {
                        'R': R,
                        'omega_rad': omega_send_rad,
                        'send_delta': bool(getattr(args, 'send_delta', False)),
                        'delta_scale': rotation_scale_val,
                        'multiply': getattr(args, 'multiply', 'left'),
                        'frame': getattr(args, 'frame', 'global'),
                        'tool_offset_z': getattr(args, 'tool_offset_z', 0.0)
                    }
                    send_allowed = not np.allclose(omega_send_rad, 0.0)
                    if not send_allowed:
                        log.info('Skipping robot request: omega zeroed by per-axis min_change_deg thresholds')
                    else:
                        log.info('queueing robot request %s', {k: req[k] for k in ('send_delta','delta_scale','multiply','frame', 'tool_offset_z')})
                        try:
                            send_fn(req)
                            log.info('request queued')
                        except Exception as e:
                            log.warning('queueing robot request failed: %s', e)
                except Exception as e:
                    log.warning('robot request failed: %s', e)
            else:
                if getattr(args, 'current_pose', None) is not None:
                    cp = np.array(args.current_pose, dtype=float)
                    if getattr(args, 'send_delta', False):
                        if np.allclose(omega_send_rad, 0.0):
                            print(
                                'DRY RUN: Skipping action because omega was zeroed by per-axis min_change_deg '
                                f'([{getattr(args, "min_change_deg_x", getattr(args, "min_change_deg", DEFAULT_MIN_CHANGE_DEG))}, '
                                f'{getattr(args, "min_change_deg_y", getattr(args, "min_change_deg", DEFAULT_MIN_CHANGE_DEG))}, '
                                f'{getattr(args, "min_change_deg_z", getattr(args, "min_change_deg", DEFAULT_MIN_CHANGE_DEG))}]deg)'
                            )
                        else:
                            rvec_delta = np.asarray(omega_send_rad, dtype=float) * float(getattr(args, 'rotation_scale', DEFAULT_ROTATION_SCALE))
                            rvec_des = np.asarray(cp[3:6], dtype=float) + rvec_delta
                            action = {
                                'abs_x': float(cp[0]),
                                'abs_y': float(cp[1]),
                                'abs_z': float(cp[2]),
                                'abs_roll': float(rvec_des[0]),
                                'abs_pitch': float(rvec_des[1]),
                                'abs_yaw': float(rvec_des[2]),
                            }
                            print('DRY RUN action:', action)
                    else:
                        if np.allclose(omega_rad, 0.0):
                            print(
                                'DRY RUN: Skipping action because omega was zeroed by per-axis min_change_deg '
                                f'([{getattr(args, "min_change_deg_x", getattr(args, "min_change_deg", DEFAULT_MIN_CHANGE_DEG))}, '
                                f'{getattr(args, "min_change_deg_y", getattr(args, "min_change_deg", DEFAULT_MIN_CHANGE_DEG))}, '
                                f'{getattr(args, "min_change_deg_z", getattr(args, "min_change_deg", DEFAULT_MIN_CHANGE_DEG))}]deg)'
                            )
                        else:
                            action = compute_action_from_pose(cp[:3], cp[3:6], R, multiply=getattr(args, 'multiply', 'left'), frame=getattr(args, 'frame', 'global'), tool_offset_z=args.tool_offset_z)
                            print('DRY RUN action:', action)
                else:
                    # print human-friendly rotation info
                    # try:
                    #     np.set_printoptions(precision=6, suppress=True)
                    #     print('Estimated rotation (deg) [wx,wy,wz]:', np.round(omega_deg, 4))
                    #     print('Rotation matrix R:\n', R)
                    # except Exception:
                    #     print('Estimated R:\n', R)
                    pass

            # visualization: show raw image, depth colormap and shear arrows
            if getattr(args, 'visualize', False):
                try:
                    vis_list = []
                    for i in range(len(sensors)):
                        img = all_images[i]
                        depth_m = all_depth_m[i]
                        shear_m = all_shear_m[i]
                        
                        if depth_m is None: continue

                        if img is None:
                            img = np.zeros((depth_m.shape[0], depth_m.shape[1], 3), dtype=np.uint8)
                        if img.dtype != np.uint8:
                            img = np.clip(img, 0, 255).astype('uint8')
                        if img.ndim == 2:
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        # depth visualization
                        depth_img = cv2.applyColorMap(np.clip(depth_m * 1000.0 * 0.25, 0, 255).astype('uint8'), cv2.COLORMAP_HOT)
                        # shear arrows
                        blank_for_arrows = np.zeros((img.shape[0], img.shape[1], 3), dtype=img.dtype)
                        shear_vis = blank_for_arrows
                        if shear_m is not None and _put_arrows_on_image is not None:
                            try:
                                shear_vis = _put_arrows_on_image(blank_for_arrows, shear_m * 50)
                            except Exception:
                                shear_vis = blank_for_arrows
                        
                        # Stack for one sensor
                        disp_one = cv2.hconcat([img, depth_img, shear_vis])
                        vis_list.append(disp_one)
                    
                    if vis_list:
                        # Stack all sensors vertically
                        display_vis = cv2.vconcat(vis_list)
                        try:
                            # Render 6DoF status in a dedicated header bar so it does not
                            # overlap/occlude the original sensor images.
                            rot_i = np.rint(np.asarray(cumulative_angle_deg, dtype=float)).astype(int)
                            trans_mm_i = np.rint(np.asarray(cumulative_translation_m, dtype=float) * 1000.0).astype(int)
                            cum_text = (
                                f'6DoF | R(deg)=[{rot_i[0]}, {rot_i[1]}, {rot_i[2]}] '
                                f'| T(mm)=[{trans_mm_i[0]}, {trans_mm_i[1]}, {trans_mm_i[2]}]'
                            )
                            header_h = 50
                            display_vis = cv2.copyMakeBorder(
                                display_vis,
                                header_h, 0, 0, 0,
                                cv2.BORDER_CONSTANT,
                                value=(30, 30, 30),
                            )
                            cv2.putText(
                                display_vis,
                                cum_text,
                                (20, 34),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 0),
                                3,
                                cv2.LINE_AA,
                            )
                            cv2.putText(
                                display_vis,
                                cum_text,
                                (20, 34),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )
                        except Exception:
                            pass
                        
                        # log.debug('Calling cv2.imshow for realtime_rot_from_shear')
                        cv2.imshow('realtime_rot_from_shear', display_vis)
                        # optionally save frames when GUI not available or user wants recordings
                        if save_dir is not None:
                            try:
                                import os
                                fname = f'frame_{frame_idx:06d}.png'
                                cv2.imwrite(os.path.join(save_dir, fname), display_vis)
                                frame_idx += 1
                            except Exception:
                                pass
                
                    # update pose visualizer if present
                    if pv is not None:
                        try:
                            # Use the same omega_rad as the terminal cumulative line (no rotation_scale).
                            # scaling is applied only when building the robot request; multiplying here made
                            # PoseVisualizer's summed angles diverge from printed cumulative_angle_deg.
                            rvec_vis = np.asarray(omega_rad, dtype=float)
                            # PoseVisualizer expects pose[:3] in mm (see pose_visualizer.PoseVisualizer.update).
                            # Drive the green translation arrow from the same cumulative T(m) as the cv2 header /
                            # terminal line (includes contact-centroid displacement when enabled).
                            _ct_mm = (np.asarray(cumulative_translation_m, dtype=float).reshape(3) * 1000.0)
                            pose_vis = [
                                float(_ct_mm[0]), float(_ct_mm[1]), float(_ct_mm[2]),
                                float(rvec_vis[0]), float(rvec_vis[1]), float(rvec_vis[2]),
                            ]
                            try:
                                should_accumulate = not np.allclose(omega_rad, 0.0)
                            except Exception:
                                should_accumulate = False
                            pv.update(
                                pose_vis,
                                accumulate=should_accumulate,
                                delta_rotation='rotvec',
                                accumulate_frame=getattr(args, 'frame', DEFAULT_FRAME),
                            )
                        except Exception:
                            pass
                    # allow key to break
                    try:
                        key = cv2.waitKey(1)
                        log.debug('cv2.waitKey returned: %s', key)
                        if key != -1:
                            if (key & 0xFF) == ord(' '):
                                cumulative_angle_deg = np.zeros(3, dtype=float)
                                cumulative_translation_m = np.zeros(3, dtype=float)
                                try:
                                    if pv is not None:
                                        pv.reset_cumulative()
                                except Exception:
                                    pass
                                print('\nCumulative rotation and translation reset to [0. 0. 0.].')
                            if (key & 0xFF) in [ord('q'), ord('Q')]:
                                log.info('Key pressed (Q), breaking loop')
                                break
                    except Exception as e:
                        log.warning('cv2.waitKey failed: %s', e)
                except Exception as e:
                    log.warning('visualization failed: %s', e)
            time.sleep(max(0.01, args.interval))
    finally:
        for s in sensors:
            try:
                if hasattr(s, 'disconnect'):
                    s.disconnect()
            except Exception:
                pass
    # stop robot worker if present
    try:
        if worker is not None:
            try:
                worker.stop()
            except Exception:
                pass
            try:
                worker.join(timeout=1.0)
            except Exception:
                pass
    except Exception:
        pass
    # cleanup cv2 windows and pose visualizer
    try:
        if 'cv2' in globals() and cv2 is not None:
            cv2.destroyAllWindows()
    except Exception:
        pass
    if pv is not None:
        try:
            pv.close()
        except Exception:
            pass
