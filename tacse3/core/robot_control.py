import numpy as np
from .math_utils import rotvec_to_matrix, matrix_to_pose, matrix_to_rotvec, clamp_rotation_step


def request_rotation(req):
    """Enforce the relative rotation limit after request gains."""
    limit = req.get('max_step_deg', 3.0)
    if req.get('send_delta', False):
        vector = np.asarray(req.get('omega_rad', np.zeros(3)), dtype=float)
        gain = float(req.get('delta_scale', .02))
        if not np.isfinite(gain):
            raise ValueError('delta_scale must be finite')
        vector = vector * gain
    elif req.get('R') is not None:
        vector = matrix_to_rotvec(req['R'])
    elif 'omega_rad' in req:
        vector = np.asarray(req['omega_rad'], dtype=float)
    else:
        raise ValueError('rotation request has neither R nor omega_rad')
    return rotvec_to_matrix(clamp_rotation_step(vector, limit))

def compute_action_from_pose(p, rvec_cur, R_est, multiply='left', frame='global', tool_offset_z=0.0):
    """
    Compute the new robot TCP pose based on the current pose, estimated rotation, and configuration.
    
    Args:
        p: Current translation [x, y, z]
        rvec_cur: Current rotation vector [rx, ry, rz]
        R_est: Estimated rotation matrix (3x3) or Delta rotation matrix
        multiply: 'left' or 'right' (legacy legacy matrix multiplication order)
        frame: 'global' (base) or 'local' (tool)
        tool_offset_z: Distance from flange to tool tip in meters
        
    Returns:
        dict: New absolute pose {abs_x, abs_y, abs_z, abs_roll, abs_pitch, abs_yaw}
    """
    # Construct 4x4 homogenous matrix for current robot TCP (flange usually)
    T_cur = np.eye(4)
    T_cur[:3, 3] = p
    T_cur[:3, :3] = rotvec_to_matrix(np.asarray(rvec_cur, dtype=float))

    # Define tool tip transformation relative to current TCP (Translation along Z)
    # If offset is 0, T_tip == T_cur
    T_offset = np.eye(4)
    T_offset[2, 3] = tool_offset_z
    
    # Calculate effective Tip Pose in Base Frame
    T_tip = T_cur @ T_offset
    
    # Rotation Matrix (3x3) to 4x4 (Rotation only, no translation)
    T_rot = np.eye(4)
    T_rot[:3, :3] = np.asarray(R_est, dtype=float)

    # Apply rotation to the Tip Frame
    if frame == 'local':
        # Local: Rotate tip in its own frame (R_new = R_cur @ R_est)
        # Position of tip stays same relative to base, as appropriate for "pivoting around tip"
        T_tip_new = T_tip @ T_rot
    elif frame == 'global':
        # Global: Rotate orientation in base frame (R_new = R_est @ R_cur)
        # We keep the tip position constant in base frame, but change its orientation
        R_tip_new = T_rot[:3, :3] @ T_tip[:3, :3]
        T_tip_new = T_tip.copy()
        T_tip_new[:3, :3] = R_tip_new
    else:
        # Fallback to legacy multiply behavior
        if multiply == 'left':
            R_des = np.asarray(R_est, dtype=float) @ T_cur[:3, :3]
        else:
            R_des = T_cur[:3, :3] @ np.asarray(R_est, dtype=float).T
        T_tip_new = T_cur.copy()
        T_tip_new[:3, :3] = R_des
        T_inv_offset = np.eye(4) # Offset handled differently in legacy

    if frame in ['local', 'global']:
        # Compute new Robot TCP (Flange) pose
        # T_tip_new = T_flange_new * T_offset
        # -> T_flange_new = T_tip_new * inv(T_offset)
        T_inv_offset = np.eye(4)
        T_inv_offset[2, 3] = -tool_offset_z
        T_new = T_tip_new @ T_inv_offset
    else:
        T_new = T_tip_new

    new_pose = matrix_to_pose(T_new)

    return {
        'abs_x': float(new_pose[0]),
        'abs_y': float(new_pose[1]),
        'abs_z': float(new_pose[2]),
        'abs_roll': float(new_pose[3]),
        'abs_pitch': float(new_pose[4]),
        'abs_yaw': float(new_pose[5]),
    }
