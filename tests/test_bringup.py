import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from tacse3.core.math_utils import matrix_to_rotvec, rotvec_to_matrix, clamp_rotation_step
from tacse3.observe import frame_metrics


class BringupTests(unittest.TestCase):
    def test_pi_and_near_pi_rotations(self):
        for axis in [np.array([1., 0, 0]), np.array([1., 2, -3]) / np.sqrt(14)]:
            for angle in [0, 1e-10, .3, np.pi - 1e-8, np.pi]:
                matrix = Rotation.from_rotvec(axis * angle).as_matrix()
                np.testing.assert_allclose(rotvec_to_matrix(matrix_to_rotvec(matrix)), matrix, atol=1e-8)

    def test_invalid_rotations_rejected(self):
        for matrix in [np.zeros((3, 3)), np.diag([1., 1, -1]), np.full((3, 3), np.nan)]:
            with self.assertRaises(ValueError):
                matrix_to_rotvec(matrix)

    def test_custom_limit_preserves_axis(self):
        vector = np.array([.3, -.4, .5])
        result = clamp_rotation_step(vector, .25)
        self.assertAlmostEqual(np.degrees(np.linalg.norm(result)), .25)
        np.testing.assert_allclose(result / np.linalg.norm(result), vector / np.linalg.norm(vector))
        np.testing.assert_equal(clamp_rotation_step(np.zeros(3), .25), np.zeros(3))

    def test_invalid_limits_and_samples_rejected(self):
        for limit in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                clamp_rotation_step([1, 0, 0], limit)
        with self.assertRaises(ValueError):
            clamp_rotation_step([np.nan, 0, 0], 1)
        with self.assertRaises(ValueError):
            frame_metrics(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2, 3)))
        with self.assertRaises(ValueError):
            frame_metrics(np.zeros((2, 2)), np.full((2, 2), np.nan), np.zeros((2, 2, 2)))

    def test_contact_and_release_metrics(self):
        image = np.zeros((2, 3), dtype=np.uint8)
        depth = np.zeros((2, 3));depth[0, 1] = .5
        self.assertEqual(frame_metrics(image, depth, np.zeros((2, 3, 2)))["depth_nonzero_pixels"], 1)
        self.assertEqual(frame_metrics(image, depth * 0, np.zeros((2, 3, 2)))["depth_nonzero_pixels"], 0)


if __name__ == "__main__":
    unittest.main()
