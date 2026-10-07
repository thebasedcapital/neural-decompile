"""Behavioral regressions for exact extraction, refusal, and certificate replay."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("automata", ROOT / "scripts/discover_automata.py")
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)
CONTAINS = ROOT / "examples/regex/contains_11.json"
MOD3 = ROOT / "examples/regex/divisible_by_3.json"


def execute(machine, bits):
    scope = {}
    exec(a.emit_python(machine), scope)
    return scope["decompiled"](bits)


class AutomataTests(unittest.TestCase):
    def test_saturation_proves_growing_recurrence_and_emits_bounded_state(self):
        result = a.extract(CONTAINS, spec="contains_11")
        self.assertEqual(result["proof"]["method"], "saturation_homomorphism")
        self.assertTrue(result["specification"]["equivalent"])
        self.assertEqual(len(result["minimal"]["outputs"]), 3)
        self.assertEqual(result["arithmetic_bridge"]["status"], "not_established")
        a.check_certificate(CONTAINS, result)
        # The source recurrence grows exponentially after 11; the finite machine
        # must not inherit that overflow risk, even on long streams.
        self.assertEqual(execute(result["minimal"], itertools.repeat(1, 100_000)), 1)
        self.assertEqual(execute(result["minimal"], itertools.chain.from_iterable(itertools.repeat((1, 0), 50_000))), 0)

    def test_exact_extraction_finds_shortest_generalization_failure(self):
        result = a.extract(MOD3, spec="divisible_by_3")
        self.assertEqual(result["specification"]["shortest_counterexample"], "10100001")
        self.assertEqual((len(result["reachable"]["states"]), len(result["minimal"]["outputs"])), (9, 7))
        a.check_certificate(MOD3, result)
        scope = {}
        exec(a.emit_python(result["minimal"]), scope)
        for length in range(8):
            for bits in itertools.product((0, 1), repeat=length):
                value = 0
                for bit in bits:
                    value = 2 * value + bit
                self.assertEqual(scope["decompiled"](bits), int(value % 3 == 0))
        self.assertEqual(scope["decompiled"]([int(v) for v in "10100001"]), 1)
        self.assertNotEqual(161 % 3, 0)

    def test_repair_deletes_one_edge_and_replays_source_to_result(self):
        original_bytes = MOD3.read_bytes()
        model, digest = a.load_model(MOD3, 0.15)
        repaired, search = a.repair_one(model, "divisible_by_3", 1, 256)
        changes = [(key, i, j) for key in ("W_hh", "W_hx", "W_y")
                   for i, row in enumerate(model[key]) for j, value in enumerate(row)
                   if value != repaired[key][i][j]]
        self.assertEqual(changes, [("W_hh", 3, 0)])
        self.assertEqual(repaired["W_hh"][3][0], 0)
        self.assertEqual(model["W_hh"][3][0], 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "repaired.json"
            path.write_text(json.dumps(repaired))
            result = a.extract(path, spec="divisible_by_3")
            self.assertTrue(result["specification"]["equivalent"])
            self.assertEqual(len(result["minimal"]["outputs"]), 3)
            self.assertEqual(result["arithmetic_bridge"]["status"], "exact_f64_and_i64")
            self.assertLess(result["arithmetic_bridge"]["absolute_operation_bound"], 2**53)
            search["original_sha256"] = digest
            result["repair_search"] = search
            a.check_certificate(path, result, original=MOD3)
            result["repair_search"]["edit"]["after"] = 99
            with self.assertRaisesRegex(ValueError, "repair search replay mismatch"):
                a.check_certificate(path, result, original=MOD3)
            bits = [1, 0, 1, 0, 0, 0, 0, 1] * 1000
            remainder = 0
            for bit in bits:
                remainder = (2 * remainder + bit) % 3
            self.assertEqual(execute(result["minimal"], bits), int(remainder == 0))
        self.assertEqual(MOD3.read_bytes(), original_bytes)

    def test_large_integer_operations_do_not_gain_a_float_exactness_claim(self):
        model = {"W_hh": [[0]], "W_hx": [[0, 0]], "b_h": [2**54],
                 "W_y": [[1], [0]], "b_y": [0, 0]}
        machine = a.enumerate_states(model, 2)
        bridge = a.arithmetic_bridge(model, machine, {"method": "exact_reachable_closure"})
        self.assertEqual(bridge["status"], "not_established")

    def test_certificate_rejects_transition_output_and_spec_tampering(self):
        result = a.extract(MOD3, spec="divisible_by_3")
        for field in ("transition", "output", "specification", "digest"):
            corrupt = copy.deepcopy(result)
            if field == "transition":
                corrupt["reachable"]["transitions"][0][1] = 0
            elif field == "output":
                corrupt["minimal"]["outputs"][0] = 0
            elif field == "specification":
                corrupt["specification"]["equivalent"] = True
            else:
                corrupt["model_sha256"] = "0" * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                a.check_certificate(MOD3, corrupt)

    def test_insufficient_abstraction_and_exploration_fail_closed(self):
        with self.assertRaises(a.Unresolved):
            a.extract(CONTAINS, limit=2, max_cap=0)
        model, _ = a.load_model(CONTAINS, 0.15)
        status, witness = a.certify_cap(model, 1, 5000)
        self.assertEqual(status, "sat")
        self.assertIn("h1", witness)
        self.assertEqual(a.certify_cap(model, 2, 5000)[0], "unsat")

    def test_quantization_strict_boundary_rounding_and_residual_refusal(self):
        model = {"W_hh": [[0]], "W_hx": [[0, 0]], "b_h": [0], "W_y": [[0], [0]], "b_y": [0, 0]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.json"
            for value, eps, expected in ((0.5, 0.51, 1), (-0.5, 0.51, -1),
                                         (2**52 + 1, 0.15, 2**52 + 1)):
                model["b_h"] = [value]
                path.write_text(json.dumps(model))
                self.assertEqual(a.load_model(path, eps)[0]["b_h"], [expected])
            model["b_h"] = [0.25]
            path.write_text(json.dumps(model))
            with self.assertRaises(a.Unresolved):
                a.load_model(path, 0.25)
            self.assertEqual(a.load_model(path, 0.25001)[0]["b_h"], [0])

    def test_first_argmax_and_behavioral_minimization(self):
        model = {"W_hh": [[0]], "W_hx": [[0, 1]], "b_h": [0], "W_y": [[0], [0]], "b_y": [0, 0]}
        machine = a.enumerate_states(model, 3)
        minimal = a.minimize(machine)
        self.assertEqual(minimal["outputs"], [0])
        self.assertEqual(minimal["transitions"], [[0, 0]])
        for bits in ([], [1], [0, 1, 1]):
            self.assertEqual(execute(minimal, bits), 0)
        with self.assertRaises(ValueError):
            execute(minimal, [2])


if __name__ == "__main__":
    unittest.main()
