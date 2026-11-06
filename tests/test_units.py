import unittest
import numpy as np
from tacse3.core.units import DEFAULT_PPMM, pixels_per_mm
from tacse3.core.estimator import pixel_xy_grid
from tacse3.core.sensor_processing import build_contact_from_daimon
from tacse3.offline_replay import metric_fields

class UnitTests(unittest.TestCase):
    def test_historical_pitch_is_mm_per_pixel(self):
        x, y = pixel_xy_grid((4, 320), DEFAULT_PPMM)
        self.assertAlmostEqual(x[0, 1] - x[0, 0], .0634e-3)
        self.assertAlmostEqual(320 / DEFAULT_PPMM, 20.288)
        self.assertAlmostEqual(pixels_per_mm(.0634), DEFAULT_PPMM)
        for value in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError): pixels_per_mm(value)
            with self.assertRaises(ValueError): pixel_xy_grid((2, 2), value)

    def test_live_preprocessing_matches_offline_nominal_units(self):
        depth = np.ones((3, 4))
        shear = np.full((3, 4, 2), 5.)
        expected = metric_fields(depth, shear, depth_unit='mm', shear_unit='dmv1-code', ppmm=DEFAULT_PPMM)
        actual = build_contact_from_daimon(depth, shear, DEFAULT_PPMM, press_increases_depth=True)
        np.testing.assert_allclose(actual[0], .001)
        np.testing.assert_allclose(actual[1], .001)
        for a, b in zip(actual[:2], expected): np.testing.assert_allclose(a, b, rtol=1e-6)
        self.assertTrue(actual[2].all())
        # Unit conversion happens once; a dimensionless gain remains separate.
        gained = build_contact_from_daimon(depth, shear, DEFAULT_PPMM, shear_scale=2.)
        np.testing.assert_allclose(gained[1], .002)
        metric = build_contact_from_daimon(depth*.001, shear*.0002, DEFAULT_PPMM,
                                          depth_in_mm=False, shear_unit='m')
        np.testing.assert_allclose(metric[1], actual[1], rtol=1e-6)
