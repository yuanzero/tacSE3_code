import unittest
import numpy as np
from tacse3.demo_pose import PosePanels
from tacse3.core.fusion import fuse_rotations
from tacse3.contact_center_translation import (
    ContactCenterTranslationState,
    fuse_plane_translation_deltas,
)


class FakePose:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.calls = []
        self.cleared = False
        self.closed = False
        self.vis = self

    def update(self, pose, **kwargs):
        self.calls.append((pose, kwargs))

    def clear_rotation(self):
        self.cleared = True

    def capture_screen_image(self, path, **kwargs):
        self.capture_path = path

    def close(self):
        self.closed = True

    def poll_events(self):
        return not self.closed


class PoseIntegrationTests(unittest.TestCase):
    def test_closed_3d_window_does_not_stop_other_panels(self):
        panels = PosePanels(["A", "B"], factory=FakePose)
        panels.panels["A"].closed = True
        row = {"display": {"rotation_vector_rad": [0, 0, 0]}}
        panels.update("A", row)
        panels.update("A", row)
        panels.update("B", row)
        self.assertNotIn("A", panels.panels)
        self.assertEqual(len(panels.panels["B"].calls), 1)

    def test_3d_uses_same_reference_and_does_not_accumulate(self):
        panels = PosePanels(["A", "B"], factory=FakePose)
        row = {"display": {"rotation_vector_rad": [0.2, -0.1, 0.3]}}
        panels.update("A", row, zero=np.array([0.1, 0, 0.2]))
        panels.update("A", row, zero=np.array([0.1, 0, 0.2]))
        for pose, kwargs in panels.panels["A"].calls:
            np.testing.assert_allclose(pose, [0, 0, 0, 0.1, -0.1, 0.1])
            self.assertEqual(kwargs, dict(accumulate=False, delta_rotation="rotvec"))
        self.assertFalse(panels.panels["A"].options["show_cumulative"])
        self.assertEqual(panels.panels["B"].calls, [])

    def test_contact_loss_capture_and_close(self):
        panels = PosePanels(["A", "B"], factory=FakePose)
        view = panels.panels["A"]
        panels.update("A", {"display": {"rotation_vector_rad": None}})
        self.assertTrue(view.cleared)
        panels.capture("/tmp", "frame", sensor_serial="A")
        self.assertTrue(view.capture_path.endswith("frame-3d-A.png"))
        self.assertFalse(hasattr(panels.panels["B"], "capture_path"))
        panels.close()
        self.assertTrue(view.closed)

    def test_fusion_and_centroid_modes(self):
        np.testing.assert_allclose(fuse_rotations([[1, 2, 3], [1, -2, -3]]), [1, 2, 3])
        np.testing.assert_allclose(fuse_plane_translation_deltas([[1, 2], [1, -2]]), [1, 2, 0])
        masks = []
        for x in range(2, 6):
            mask = np.zeros((10, 10), bool)
            mask[4:6, x : x + 2] = True
            masks.append(mask)
        for mode, expected in [("consecutive", 0.0001), ("window", 0.0002)]:
            state = ContactCenterTranslationState(1, mode=mode, smooth_window=2)
            for mask in masks:
                delta = state.update([mask], 10.0)
            np.testing.assert_allclose(delta, [expected, 0, 0], atol=1e-10)
            state.update([np.zeros((10, 10), bool)], 10.0)
            np.testing.assert_allclose(state.update([masks[0]], 10.0), 0)

    def test_arrow_geometry_in_both_z_directions(self):
        try:
            from tacse3.pose_visualizer import arrow_mesh
        except ImportError:
            self.skipTest("Optional Open3D is not installed")
        for sign in [1, -1]:
            vertices = np.asarray(arrow_mesh([0, 0, 0], [0, 0, sign * 0.02]).vertices)
            self.assertAlmostEqual(np.max(vertices[:, 2] * sign), 0.02, places=7)
            self.assertAlmostEqual(np.min(vertices[:, 2] * sign), 0, places=7)
