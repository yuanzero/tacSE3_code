import numpy as np

def fuse_rotations(rotations_rad: list, fusion_signs=None) -> np.ndarray:
    """
    Fuse multiple rotation vectors from different sensors.
    
    Args:
        rotations_rad: List of [rx, ry, rz] vectors (radians)
        fusion_signs: Optional [sx, sy, sz] signs for the second sensor (e.g. [1, -1, -1])
        
    Returns:
        Fused rotation vector [rx, ry, rz]
    """
    if not rotations_rad:
        return np.zeros(3)

    num_sensors = len(rotations_rad)

    if num_sensors == 2:
        # 针对两个传感器的镜像融合策略
        # "两个是先减再平均，有一个是相加再平均"
        # 假设 Sensor 1 (rotations_rad[1]) 相对于 Sensor 0 (rotations_rad[0]) 是镜像的
        
        r0 = np.asarray(rotations_rad[0], dtype=float)
        r1 = np.asarray(rotations_rad[1], dtype=float)
        
        # 默认融合系数: Y轴同向(1.0)，X和Z轴反向(-1.0)
        if fusion_signs is None:
            signs = np.array([1.0, -1.0, -1.0])
        else:
            signs = np.asarray(fusion_signs, dtype=float)
        
        # Logic: 
        # For sign -1 (mirrored): real_rot = (r0 - r1) / 2
        # For sign +1 (aligned):  real_rot = (r0 + r1) / 2
        # This can be written as (r0 + sign * r1) / 2
        
        omega_rad = (r0 + r1 * signs) / 2.0
        return omega_rad
        
    else:
        # Default average strategy for N sensors
        stack = np.stack(rotations_rad, axis=0)
        return np.mean(stack, axis=0)
