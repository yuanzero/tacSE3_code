import numpy as np
import open3d as o3d

from .core.math_utils import euler_angles_to_rotation_matrix, rotvec_to_matrix


def arrow_mesh(origin, vector, cylinder_radius=0.002, cone_radius=0.004, cylinder_length_ratio=0.8, resolution=20):
    v = np.array(vector, dtype=float)
    length = np.linalg.norm(v)
    if length == 0:
        # return a tiny sphere to indicate zero-length
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        sph.translate(origin)
        sph.compute_vertex_normals()
        return sph
    direction = v / length
    cyl_len = length * cylinder_length_ratio
    cone_len = length - cyl_len
    mesh_cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=cyl_len, resolution=resolution, split=4)
    mesh_cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=cone_len, resolution=resolution, split=1)
    mesh_cyl.translate((0, 0, cyl_len / 2.0))
    mesh_cone.translate((0, 0, cyl_len + cone_len / 2.0))
    arrow = mesh_cyl + mesh_cone
    arrow.compute_vertex_normals()
    z_axis = np.array([0.0, 0.0, 1.0])
    if not np.allclose(direction, z_axis):
        rot_axis = np.cross(z_axis, direction)
        rot_axis_norm = np.linalg.norm(rot_axis)
        if rot_axis_norm < 1e-8:
            R = np.eye(3)
        else:
            rot_axis = rot_axis / rot_axis_norm
            angle = np.arccos(np.clip(z_axis @ direction, -1.0, 1.0))
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * angle)
        arrow.rotate(R, center=(0, 0, 0))
    arrow.translate(origin)
    return arrow


class PoseVisualizer:
    """Simple Open3D visualizer for a 6DOF pose.

    Pose format: [x(mm), y(mm), z(mm), rx(deg), ry(deg), rz(deg)]
    Rotation is shown as three small colored arrows indicating the rotated local axes.
    Translation is intentionally ignored in visualization.

    By default the visualized arrows use a fixed physical length (meters) and
    do not scale with the magnitude of the pose translation.
    """

    def __init__(
        self,
        window_name="Pose Visualizer",
        trans_length=0.02,
        axis_length=0.02,
        width=1600,
        height=1600,
        show_angular_velocity=False,
        show_cumulative=True,
        min_change_deg=0.0,
        axis_mapping=("x", "y", "z"),
    ):
        """Create visualizer.

        :param trans_length: kept for backward compatibility (unused)
        :param axis_length: fixed length (meters) used to draw the orientation axes
        :param width: window width in pixels
        :param height: window height in pixels
        :param show_angular_velocity: whether to open the angular-velocity Open3D window
        :param show_cumulative: whether to open the cumulative-angle Open3D window
        :param axis_mapping: how input xyz maps to visualized XYZ axes.
                             Supports optional +/- sign, e.g. ("z", "x", "-y")
                             means input z->visual +X, input x->visual +Y,
                             input y->visual -Z.
        """
        self.show_angular_velocity = bool(show_angular_velocity)
        self.show_cumulative = bool(show_cumulative)

        self.angular_vis = None
        self._geoms = {}
        if self.show_angular_velocity:
            self.angular_vis = self._create_visualizer(f"{window_name} - Angular Velocity", width, height)
            self._add_frame(self.angular_vis, self._geoms)

        self.cumulative_vis = None
        self._cumulative_geoms = {}
        if self.show_cumulative:
            self.cumulative_vis = self._create_visualizer(f"{window_name} - Cumulative Angle", width, height)
            self._add_frame(self.cumulative_vis, self._cumulative_geoms)

        # Keep backward compatibility for external code that expects `self.vis`.
        if self.angular_vis is not None:
            self.vis = self.angular_vis
        elif self.cumulative_vis is not None:
            self.vis = self.cumulative_vis
        else:
            self.vis = self._create_visualizer(window_name, width, height)
            self._add_frame(self.vis, self._geoms)

        # fixed arrow lengths (meters)
        self.trans_length = float(trans_length)
        self.axis_length = float(axis_length)
        self._axis_indices = self._parse_axis_mapping(axis_mapping)
        # cumulative rotation tracking (degrees)
        min_change_arr = np.asarray(min_change_deg, dtype=float).reshape(-1)
        if min_change_arr.size == 1:
            min_change_arr = np.repeat(min_change_arr, 3)
        elif min_change_arr.size != 3:
            raise ValueError("min_change_deg must be a scalar or length-3 iterable")
        self.min_change_deg = min_change_arr
        self._cum_angles = np.zeros(3, dtype=float)
        # Keep for backward compatibility; cumulative display now uses summed Euler
        # angles for both euler and rotvec delta modes to match terminal accumulation.
        self._cum_R = np.eye(3, dtype=float)

    @staticmethod
    def _parse_axis_mapping(axis_mapping):
        """Convert signed axis labels into source indices/signs for xyz data."""
        if len(axis_mapping) != 3:
            raise ValueError("axis_mapping must have exactly 3 entries")
        lut = {"x": 0, "y": 1, "z": 2}
        src_indices = []
        src_signs = []
        for axis in axis_mapping:
            key = str(axis).strip().lower()
            sign = 1.0
            if key.startswith("+"):
                key = key[1:]
            elif key.startswith("-"):
                sign = -1.0
                key = key[1:]
            if key not in lut:
                raise ValueError("axis_mapping entries must be one of: x, y, z (optionally prefixed by +/-)")
            src_indices.append(lut[key])
            src_signs.append(sign)
        if len(set(src_indices)) != 3:
            raise ValueError("axis_mapping must contain x, y, z exactly once (ignoring sign)")
        return tuple(src_indices), np.asarray(src_signs, dtype=float)

    def _remap_xyz(self, xyz):
        arr = np.asarray(xyz, dtype=float).reshape(3)
        src_indices, src_signs = self._axis_indices
        return arr[list(src_indices)] * src_signs

    def _create_visualizer(self, window_name, width, height):
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name, width=width, height=height)
        return vis

    def _add_frame(self, vis, geoms):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
        vis.add_geometry(frame)
        geoms['frame'] = frame

    def _remove_geometry(self, vis, geoms, key):
        if key in geoms:
            vis.remove_geometry(geoms[key], reset_bounding_box=False)
            del geoms[key]

    def _update_axes(self, vis, geoms, prefix, origin, rotation_matrix, colors):
        axes = np.eye(3) * self.axis_length
        for i in range(3):
            self._remove_geometry(vis, geoms, f'{prefix}{i}')
        for i in range(3):
            vec = rotation_matrix @ axes[i]
            arrow = arrow_mesh(origin, vec, cylinder_radius=0.002, cone_radius=0.004)
            arrow.paint_uniform_color(colors[i])
            vis.add_geometry(arrow)
            geoms[f'{prefix}{i}'] = arrow

    def _pose_to_transform(self, pose):
        """Build (t, R) from pose [mm, mm, mm, ...].

        Default Euler mode: pose[3:6] are rx,ry,rz in degrees, same as
        ``euler_angles_to_rotation_matrix`` / ``euler_degrees_to_matrix``:
        R = Rz @ Ry @ Rx (Tait–Bryan / extrinsic XYZ).
        """
        t = self._remap_xyz(pose[:3]) / 1000.0  # mm -> m
        euler_deg = self._remap_xyz(pose[3:6])
        R = euler_angles_to_rotation_matrix(np.deg2rad(euler_deg))
        return t, R

    def update(self, pose, accumulate=None, delta_rotation='euler', accumulate_frame='global'):
        """Update the visualized pose. pose: iterable length 6.

        ``delta_rotation``:
          - ``'euler'`` (default): pose[3:6] are rx,ry,rz in **degrees**; R follows
            ``euler_angles_to_rotation_matrix`` (Rz @ Ry @ Rx).
          - ``'rotvec'``: pose[3:6] are a rotation vector in **radians**, same as
            ``RobotWorker`` when ``send_delta=True`` (``rotvec_to_matrix(rvec)``).

        If `self.show_cumulative` is True (or `accumulate=True` is passed), the
        pose's rotation is composed into the cumulative view by summing per-axis
        increments in degrees and rebuilding an Euler matrix. In rotvec mode this
        intentionally matches terminal accumulation semantics (sum of xyz deltas),
        rather than matrix composition.
        """
        if accumulate is None:
            accumulate = self.show_cumulative

        if delta_rotation == 'rotvec':
            t = self._remap_xyz(pose[:3]) / 1000.0
            R = rotvec_to_matrix(self._remap_xyz(pose[3:6]))
        else:
            t, R = self._pose_to_transform(pose)

        if self.angular_vis is not None:
            self._update_axes(self.angular_vis, self._geoms, 'ax', np.zeros(3, dtype=float), R, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        elif self.cumulative_vis is None:
            # Fallback single window when both toggles are disabled.
            self._update_axes(self.vis, self._geoms, 'ax', np.zeros(3, dtype=float), R, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        try:
            if accumulate and self.cumulative_vis is not None:
                if delta_rotation == 'rotvec':
                    inc_deg = np.degrees(self._remap_xyz(pose[3:6]))
                    self._cum_angles += inc_deg
                    Rcum = euler_angles_to_rotation_matrix(np.deg2rad(self._cum_angles.astype(float)))
                else:
                    inc = self._remap_xyz(pose[3:6])
                    self._cum_angles += inc
                    Rcum = euler_angles_to_rotation_matrix(np.deg2rad(self._cum_angles.astype(float)))
                self._update_axes(
                    self.cumulative_vis,
                    self._cumulative_geoms,
                    'cax',
                    np.zeros(3, dtype=float),
                    Rcum,
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                )
        except Exception:
            # don't let visualization errors break the main loop
            pass
        if self.angular_vis is not None:
            self.angular_vis.poll_events()
            self.angular_vis.update_renderer()
        elif self.cumulative_vis is None:
            self.vis.poll_events()
            self.vis.update_renderer()
        if self.cumulative_vis is not None:
            self.cumulative_vis.poll_events()
            self.cumulative_vis.update_renderer()

    def close(self):
        if self.angular_vis is not None:
            self.angular_vis.destroy_window()
        elif self.cumulative_vis is None:
            self.vis.destroy_window()
        if self.cumulative_vis is not None:
            self.cumulative_vis.destroy_window()

    def reset_cumulative(self):
        """Reset cumulative rotation to zero and remove cumulative axes from the scene."""
        try:
            self._cum_angles = np.zeros(3, dtype=float)
            self._cum_R = np.eye(3, dtype=float)
            if self.cumulative_vis is not None:
                for k in ['cax0', 'cax1', 'cax2']:
                    try:
                        self._remove_geometry(self.cumulative_vis, self._cumulative_geoms, k)
                    except Exception:
                        pass
                self.cumulative_vis.poll_events()
                self.cumulative_vis.update_renderer()
        except Exception:
            pass


if __name__ == '__main__':
    # demo: animate a few poses
    vis = PoseVisualizer()
    try:
        for k in range(40):
            x = 20.0 * np.sin(0.2 * k)
            y = 10.0 * np.cos(0.15 * k)
            z = 5.0 * np.sin(0.1 * k)
            rx = 10.0 * np.sin(0.05 * k)
            ry = 8.0 * np.cos(0.07 * k)
            rz = 15.0 * np.sin(0.03 * k)
            pose = [x, y, z, rx, ry, rz]
            vis.update(pose)
    finally:
        vis.close()
