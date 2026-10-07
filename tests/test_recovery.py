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

    def test_blend_patch_reconstructs_average_of_parts(self):
        rng = np.random.default_rng(412)
        channels, outputs = 6, 7
        decoder = m.Decoder(
            rng.normal(size=channels), np.linspace(.5, 2., channels),
            rng.normal(size=outputs), np.linspace(.3, 3., outputs),
            rng.normal(size=(channels + 1, outputs)),
            rng.normal(size=(3, channels + 1, outputs)),
        )
        raw = rng.normal(size=(300, channels)) * decoder.scale + decoder.mean
        targets = decoder.predict(raw)
        fitted, count, _, patch = m.adapt(decoder, raw, targets, raw, m.BLEND_METHOD, 10.)
        self.assertEqual(count, decoder.weights.size)
        restored = m.apply_patch(decoder, m.BLEND_METHOD, patch)
        np.testing.assert_allclose(restored, fitted, rtol=1e-12, atol=1e-12)
        recentered = m.adapt(decoder, raw, targets, raw, "recenter", 0.)[0]
        repaired = m.adapt(decoder, raw, targets, raw, m.REPAIR_METHOD, 10.)[0]
        np.testing.assert_allclose(fitted, 0.5 * (recentered + repaired), rtol=1e-12, atol=1e-12)
        with self.assertRaises(ValueError):
            m.apply_patch(decoder, m.BLEND_METHOD, patch[:-1])

    def test_alpha_overrides_are_targeted_and_recorded(self):
        choices = {"three_source_separate_bias_rank_6_repair": {"30": 1.0, "45": 1.0},
                   "anchored": {"30": 1.0}}
        protocol = {"reused_data_alpha_overrides": {"three_source_separate_bias_rank_6_repair": {"30": 30.0}}}
        applied = ex.apply_alpha_overrides(protocol, choices)
        self.assertEqual(choices["three_source_separate_bias_rank_6_repair"]["30"], 30.0)
        self.assertEqual(choices["three_source_separate_bias_rank_6_repair"]["45"], 1.0)
        self.assertEqual(choices["anchored"]["30"], 1.0)
        self.assertEqual(applied, {"three_source_separate_bias_rank_6_repair@30s": 30.0})
        with self.assertRaises(ValueError):
            ex.apply_alpha_overrides({"reused_data_alpha_overrides": {"missing_method": {"30": 1.}}}, choices)
        with self.assertRaises(ValueError):
            ex.apply_alpha_overrides({"reused_data_alpha_overrides": {"anchored": {"99": 1.}}}, choices)


if __name__ == "__main__":
    unittest.main()
