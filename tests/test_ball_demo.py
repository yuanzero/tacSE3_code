import unittest
import contextlib
import io
import tempfile
from pathlib import Path
import numpy as np
from tacse3.ball_demo import main, response_metrics, render_panel
from tacse3.offline_replay import replay
from tacse3.core.units import DEFAULT_PPMM


class BallDemoTests(unittest.TestCase):
    def test_existing_session_is_rejected_without_changing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            path.write_text('{"original": true}', encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(["--sensor-serial", "TEST", "--output", directory])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(path.read_text(encoding="utf-8"), '{"original": true}')
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_invalid_arguments_fail_before_opening_devices(self):
        for argv in [
            ["--sensor-serial", "A", "A"],
            ["--sensor-serial", "../A"],
            ["--sensor-serial", "A", "--mm-per-pixel", "0"],
        ]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(argv)
            self.assertEqual(error.exception.code, 2)

    def test_light_response_is_visible_below_estimator_threshold(self):
        depth = np.full((24, 32), 0.1)
        shear = np.zeros((24, 32, 2))
        shear[..., 0] = 5
        result = replay(depth, shear, depth_unit="mm", shear_unit="dmv1-code", ppmm=DEFAULT_PPMM)
        metrics = response_metrics(depth, shear)
        self.assertEqual(result["status"], "insufficient_contact")
        self.assertEqual(metrics["response_pixels"], depth.size)
        self.assertAlmostEqual(metrics["shear_peak_nominal_mm"], 1.0)

    def test_no_depth_does_not_save_shear_noise_as_contact_peak(self):
        metrics = response_metrics(np.zeros((2, 2)), np.ones((2, 2, 2)) * 100)
        self.assertEqual(metrics["shear_peak_nominal_mm"], 0)
        self.assertEqual(metrics["response_pixels"], 0)

    def test_render_with_history_gaps_and_display_zero(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV visualization dependency is not installed")
        depth = np.full((24, 32), 0.1)
        shear = np.zeros((24, 32, 2))
        result = replay(depth, shear, depth_unit="mm", shear_unit="dmv1-code", ppmm=DEFAULT_PPMM)
        history = [(0.0, np.array([0.01, 0, 0])), (1.0, np.full(3, np.nan)), (2.0, np.zeros(3))]
        panel = render_panel(
            np.zeros((24, 32), np.uint8),
            depth,
            shear,
            result,
            "test",
            zero=np.array([0.01, 0, 0]),
            history=history,
        )
        self.assertEqual(panel.shape, (505, 960, 3))
        self.assertEqual(panel.dtype, np.uint8)
