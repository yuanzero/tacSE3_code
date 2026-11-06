"""Bridge the existing Open3D visualizer to per-sensor demo display vectors."""

from pathlib import Path
import numpy as np


class PosePanels:
    """One relative coordinate frame per sensor, without cumulative integration."""

    def __init__(self, serials, factory=None):
        if factory is None:
            from .pose_visualizer import PoseVisualizer

            factory = PoseVisualizer
        self.panels = {}
        try:
            for index, serial in enumerate(serials):
                panel = factory(
                    window_name=f"TacSE3 3D | {serial}",
                    width=520,
                    height=480,
                    show_angular_velocity=False,
                    show_cumulative=False,
                    left=1050 + (index // 2) * 550,
                    top=60 + (index % 2) * 510,
                )
                self.panels[serial] = panel
                # A grey reference frame distinguishes it from the live RGB axes.
                if hasattr(panel, "_geoms"):
                    panel._geoms["frame"].paint_uniform_color([0.5, 0.5, 0.5])
                    panel.vis.update_geometry(panel._geoms["frame"])
        except Exception:
            self.close()
            raise

    def update(self, serial, result, zero=None):
        vector = result["display"]["rotation_vector_rad"]
        panel = self.panels.get(serial)
        if panel is None:
            return
        if not panel.vis.poll_events():
            panel.close()
            del self.panels[serial]
            return
        if vector is None:
            panel.clear_rotation()
            return
        vector = np.asarray(vector) - (np.zeros(3) if zero is None else zero)
        panel.update([0.0, 0.0, 0.0, *vector], accumulate=False, delta_rotation="rotvec")

    def capture(self, directory, prefix, sensor_serial=None):
        for serial, panel in self.panels.items():
            if sensor_serial is not None and serial != sensor_serial:
                continue
            panel.vis.capture_screen_image(
                str(Path(directory) / f"{prefix}-3d-{serial}.png"), do_render=True
            )

    def close(self):
        for panel in self.panels.values():
            panel.close()
        self.panels.clear()
