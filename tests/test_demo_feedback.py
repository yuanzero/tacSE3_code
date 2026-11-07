import copy
import unittest
import numpy as np

from tacse3.demo_feedback import DisplayFeedback, FeedbackConfig


def sample(timestamp, vector=(0.0, 0.0, 0.0), pixels=1000):
    return dict(time=timestamp, rotation_vector_rad=list(vector), rank=5, contact_pixels=pixels)


class DisplayFeedbackTests(unittest.TestCase):
    def test_raw_is_unchanged_and_isolated_spike_is_removed(self):
        state = DisplayFeedback(FeedbackConfig(activation_ms=0))
        for i in range(5):
            state.update(sample(i * 0.05))
        row = sample(0.25, (0.5, 0, 0))
        original = copy.deepcopy(row)
        result = state.update(row)
        self.assertEqual(row, original)
        np.testing.assert_array_equal(result["rotation_vector_rad"], [0, 0, 0])

    def test_loss_and_small_contact_clear_history(self):
        state = DisplayFeedback(FeedbackConfig(activation_ms=0))
        state.update(sample(0, (0.1, 0, 0)))
        self.assertEqual(state.update(sample(0.05, pixels=30))["state"], "small_contact")
        self.assertEqual(
            state.update(dict(time=0.1, response_pixels=200))["state"], "light_response"
        )
        np.testing.assert_allclose(
            state.update(sample(0.2, (-0.2, 0, 0)))["rotation_vector_rad"], [-0.2, 0, 0]
        )

    def test_activation_and_clock_gap(self):
        state = DisplayFeedback()
        self.assertEqual(state.update(sample(0))["state"], "settling")
        self.assertEqual(state.update(sample(0.05))["state"], "settling")
        self.assertEqual(state.update(sample(0.1))["state"], "tracking")
        self.assertEqual(state.update(sample(1))["state"], "settling")
        self.assertEqual(state.update(sample(0.9))["state"], "settling")

    def test_step_converges_and_zero_smoothing_is_identity(self):
        for interval in [0.02, 0.05]:
            state = DisplayFeedback(FeedbackConfig(activation_ms=0))
            for t in np.arange(0, 0.2, interval):
                state.update(sample(t))
            for t in np.arange(0.2, 1.0, interval):
                value = state.update(sample(t, (0.1, -0.1, 0.03)))["rotation_vector_rad"]
            np.testing.assert_allclose(value, [0.1, -0.1, 0.03], atol=0.001)
        state = DisplayFeedback(FeedbackConfig(smoothing_ms=0, activation_ms=0))
        for i in range(5):
            np.testing.assert_allclose(
                state.update(sample(i * 0.05, (i, 0, 0)))["rotation_vector_rad"], [i, 0, 0]
            )

    def test_invalid_configuration_and_estimates(self):
        for kwargs in [
            dict(smoothing_ms=-1),
            dict(activation_ms=float("nan")),
            dict(min_contact_pixels=2),
            dict(max_gap_s=0),
        ]:
            with self.assertRaises(ValueError):
                FeedbackConfig(**kwargs)
        row = sample(0, (float("nan"), 0, 0))
        self.assertEqual(DisplayFeedback().update(row)["state"], "invalid_estimate")
