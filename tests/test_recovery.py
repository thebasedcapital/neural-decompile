"""Consumer-visible causality, metric, and compact-patch invariants."""
import importlib.util
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("recovery_model", ROOT / "scripts/recovery/model.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
ESPEC = importlib.util.spec_from_file_location("recovery_experiment", ROOT / "scripts/recovery/experiment.py")
ex = importlib.util.module_from_spec(ESPEC)
sys.path.insert(0, str(ROOT / "scripts/recovery"))
ESPEC.loader.exec_module(ex)


class RecoveryTests(unittest.TestCase):
    def test_future_spikes_cannot_change_past_predictions_and_filter_expires(self):
        counts = np.zeros((40, 2))
        counts[0, 0] = 1
        changed = counts.copy()
        changed[20:, :] = 1000
        original = m.causal_features(counts)
        np.testing.assert_array_equal(original[:20], m.causal_features(changed)[:20])
        np.testing.assert_array_equal(original[:9], m.causal_features(counts[:9]))
        np.testing.assert_array_equal(original[12:], 0)
        self.assertAlmostEqual(original[:, 0].sum(), 1)

    def test_constant_outputs_have_zero_weight_in_variance_weighted_r2(self):
        truth = np.array([[0., 3.], [1., 3.], [2., 3.]])
        prediction = np.array([[0., 99.], [0., 99.], [0., 99.]])
        overall, dimensions = m.score(truth, prediction)
        self.assertAlmostEqual(overall, -1.5)
        self.assertEqual(dimensions, [-1.5, 0.])

    def test_compact_repairs_recover_known_changes_and_fold_without_normalization_loss(self):
        rng = np.random.default_rng(72)
        c, o = 4, 7
        decoder = m.Decoder(np.array([2., -1., 0.2, 9.]), np.array([3., 2., 4., 0.5]),
                            np.arange(o) / 10, np.arange(1, o + 1) / 3,
                            rng.normal(size=(c + 1, o)), rng.normal(size=(3, c + 1, o)))
        raw = rng.normal(size=(200, c)) * decoder.scale + decoder.mean
        unseen = rng.normal(size=(40, c)) * decoder.scale + decoder.mean
        cases = {
            "gain_repair": np.r_[np.array([0.4, -0.2, 0.7, -0.3]), np.arange(o) / 10],
            "drift_basis": np.array([0.2, -0.6, 0.4]),
        }
        for method, patch in cases.items():
            with self.subTest(method=method):
                expected_w = decoder.weights.copy()
                if method == "gain_repair":
                    expected_w[:-1] *= 1 + patch[:c, None]
                    expected_w[-1] += patch[c:]
                else:
                    for coefficient, basis in zip(patch, decoder.basis):
                        expected_w += coefficient * basis
                targets = decoder.predict(raw, expected_w)
                fitted, _, _, serialized_patch = m.adapt(decoder, raw, targets, raw, method, 1e-12)
                restored = m.apply_patch(decoder, method, serialized_patch)
                slope, bias = decoder.folded(restored)
                np.testing.assert_allclose(unseen @ slope + bias, decoder.predict(unseen, expected_w), atol=1e-9)
                np.testing.assert_allclose(decoder.predict(unseen, fitted), decoder.predict(unseen, expected_w), atol=1e-9)
                with self.assertRaises(ValueError):
                    m.apply_patch(decoder, method, [float("nan")])

    def test_rank_six_repair_recovers_rotation_and_independent_bias(self):
        rng = np.random.default_rng(931)
        channels, outputs, rank = 8, 7, 6
        decoder = m.Decoder(
            rng.normal(size=channels), np.linspace(.3, 2., channels),
            rng.normal(size=outputs), np.linspace(.2, 3., outputs),
            rng.normal(size=(channels + 1, outputs)),
            rng.normal(size=(4, channels + 1, outputs)),
        )
        decoder.basis[0] *= 100
        expected = decoder.weights + 2 * decoder.basis[-3:].mean(axis=0)
        expected[:-1] += rng.normal(size=(channels, rank)) @ rng.normal(size=(rank, outputs))
        expected[-1] += rng.normal(size=outputs)
        raw = rng.normal(size=(400, channels)) * decoder.scale + decoder.mean
        unseen = rng.normal(size=(40, channels)) * decoder.scale + decoder.mean
        targets = decoder.predict(raw, expected)
        method = "three_source_separate_bias_rank_6_repair"
        fitted, _, _, patch = m.adapt(decoder, raw, targets, raw, method, 1e-12)
        restored = m.apply_patch(decoder, method, patch)
        slope, bias = decoder.folded(restored)
        np.testing.assert_allclose(unseen @ slope + bias, decoder.predict(unseen, expected), atol=1e-8)
        np.testing.assert_allclose(decoder.predict(unseen, fitted), decoder.predict(unseen, expected), atol=1e-8)
        with self.assertRaises(ValueError):
            m.apply_patch(decoder, method, patch[:-1])
        for invalid in ([float("nan")], [patch], [[]]):
            with self.subTest(patch=invalid[:1]), self.assertRaises(ValueError):
                m.apply_patch(decoder, method, invalid)
        with self.assertRaises(ValueError):
            m.apply_patch(decoder, "unknown", [])
        empty = raw[:0]
        unchanged, count, _, empty_patch = m.adapt(decoder, empty, targets[:0], raw, method, 1.)
        np.testing.assert_array_equal(unchanged, decoder.weights)
        np.testing.assert_array_equal(m.apply_patch(decoder, method, empty_patch), decoder.weights)
        self.assertEqual(count, 0)
        decoder.basis = decoder.basis[:2]
        with self.assertRaises(ValueError):
            m.apply_patch(decoder, method, patch)

    def test_causal_smoothing_ignores_future_bins_and_resets(self):
        values = np.arange(20, dtype=float).reshape(10, 2)
        original = m.smooth(values, 0.1)
        changed = values.copy()
        changed[5:] += 100.0
        np.testing.assert_array_equal(m.smooth(changed, 0.1)[:5], original[:5])
        np.testing.assert_array_equal(m.smooth(values[:1], 0.1), values[:1])
        with self.assertRaises(ValueError):
            m.smooth(values, 0.0)
        with self.assertRaises(ValueError):
            m.smooth(values, 1.5)

    def test_running_zscore_is_causal_and_forgets_a_baseline_shift(self):
        rng = np.random.default_rng(5)
        features = rng.normal(size=(6000, 3))
        features[3000:] += 50.0  # recording-level offset, e.g. electrode drift
        mean, variance = m.prefix_statistics(features[:1500])
        z = m.running_zscore(features, mean, variance, tau_s=10., bin_ms=20)
        changed = features.copy()
        changed[4000:] -= 999.0
        np.testing.assert_array_equal(m.running_zscore(changed, mean, variance, 10., 20)[:4000], z[:4000])
        self.assertGreater(np.abs(z[3000]).min(), 10.)  # the shift is visible at first...
        self.assertLess(np.abs(z[-1000:].mean(axis=0)).max(), 0.2)  # ...then absorbed after ~5 tau
        with self.assertRaises(ValueError):
            m.running_zscore(features, mean, variance, tau_s=0.02, bin_ms=20)

    def test_silent_prefix_channel_cannot_explode_when_it_wakes(self):
        features = np.zeros((200, 2))
        features[:, 1] = np.arange(200) % 2
        mean, variance = m.prefix_statistics(features[:100])
        self.assertEqual(variance[0], 1.0)
        features[150:, 0] = 1.0
        z = m.running_zscore(features, mean, variance, tau_s=30., bin_ms=20)
        self.assertLess(np.abs(z[:, 0]).max(), 2.0)


if __name__ == "__main__":
    unittest.main()
