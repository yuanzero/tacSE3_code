"""TacSE3 — realtime tactile sensing and SE(3) motion estimation for UR5 (LeRobot).

Install: ``pip install -e .`` from the ``TacSE3`` project directory.

Run: ``python -m tacse3.run_monitor --help``
"""

from importlib import import_module

__all__ = [
    "add_primary_cli_args",
    "simple_sensor_loop_cli",
    "DEFAULT_ANGLE_SCALE",
    "DEFAULT_ANGLE_SCALE_X",
    "DEFAULT_ANGLE_SCALE_Y",
    "DEFAULT_ANGLE_SCALE_Z",
    "DEFAULT_AXIS_ORDER",
    "DEFAULT_AXIS_SIGN",
    "DEFAULT_CONTACT_CENTER_TRANSLATION",
    "DEFAULT_CONTACT_CENTER_TRANSLATION_MODE",
    "DEFAULT_CONTACT_THRESH_MM",
    "DEFAULT_DURATION",
    "DEFAULT_FRAME",
    "DEFAULT_INTERVAL",
    "DEFAULT_MAX_STEP_DEG",
    "DEFAULT_MIN_CHANGE_DEG",
    "DEFAULT_MIN_CHANGE_DEG_X",
    "DEFAULT_MIN_CHANGE_DEG_Y",
    "DEFAULT_MIN_CHANGE_DEG_Z",
    "DEFAULT_PPMM",
    "DEFAULT_ROTATION_SCALE",
    "DEFAULT_SMOOTH_WINDOW",
    "DEFAULT_TOOL_OFFSET_Z",
]


def __getattr__(name):
    if name in __all__:
        return getattr(import_module('.sensor_loop', __name__), name)
    raise AttributeError(name)
