import numpy as np
from scipy.spatial.transform import Rotation


def clamp_rotation_step(vector, max_step_deg):
    """Limit a finite rotation vector without changing its direction."""
    vector = np.asarray(vector, dtype=float)
    limit = float(max_step_deg)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("rotation must be a finite 3-vector")
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("max_step_deg must be positive and finite")
    norm = float(np.linalg.norm(vector))
    return vector * min(1.0, np.radians(limit) / norm) if norm else vector.copy()

def rotvec_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert rotation vector to rotation matrix."""
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)
    k = (rvec / theta).reshape(3)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

def matrix_to_rotvec(Rm: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to rotation vector."""
    matrix = np.asarray(Rm, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation matrix must be finite and 3x3")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-6):
        raise ValueError("matrix must be a proper rotation")
    return Rotation.from_matrix(matrix).as_rotvec()

def euler_degrees_to_matrix(euler_deg: list) -> np.ndarray:
    """Convert [x, y, z] Euler angles in degrees to a rotation matrix (extrinsic XYZ)."""
    rx, ry, rz = np.radians(euler_deg)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

def matrix_to_pose(T: np.ndarray) -> np.ndarray:
    """Convert 4x4 matrix to 6D pose [x,y,z,rx,ry,rz]."""
    pose = np.zeros(6)
    pose[:3] = T[:3, 3]
    pose[3:] = matrix_to_rotvec(T[:3, :3])
    return pose

def parse_matrix_arg(s: str) -> np.ndarray:
    """Parse a comma/space-separated 9-number string into 3x3 matrix."""
    parts = [p for p in s.replace(",", " ").split() if p]
    if len(parts) != 9:
        raise ValueError("Matrix must contain 9 numbers")
    vals = list(map(float, parts))
    return np.array(vals, dtype=float).reshape((3, 3))

def euler_angles_to_rotation_matrix(angles_rad):
    rx, ry, rz = float(angles_rad[0]), float(angles_rad[1]), float(angles_rad[2])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, cx, -sx],
                   [0.0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0.0, sy],
                   [0.0, 1.0, 0.0],
                   [-sy, 0.0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0.0],
                   [sz, cz, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)

    return Rz @ Ry @ Rx
