from importlib import import_module

_exports = {'DaimonMonitor': 'monitor', 'RobotWorker': 'robot_worker',
            'estimate_rotation_from_shear': 'estimator',
            'build_contact_from_daimon': 'sensor_processing', 'fuse_rotations': 'fusion'}


def __getattr__(name):
    if name in _exports:
        return getattr(import_module('.' + _exports[name], __name__), name)
    math_names = {'rotvec_to_matrix', 'matrix_to_rotvec', 'matrix_to_pose',
                  'euler_degrees_to_matrix', 'parse_matrix_arg',
                  'euler_angles_to_rotation_matrix', 'clamp_rotation_step'}
    if name in math_names:
        return getattr(import_module('.math_utils', __name__), name)
    raise AttributeError(name)

__all__ = [
    "DaimonMonitor",
    "RobotWorker", 
    "estimate_rotation_from_shear",
    "build_contact_from_daimon",
    "fuse_rotations"
]
