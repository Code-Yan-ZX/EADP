"""CPU-only invariants for the preregistered native-token geometry."""

from collections import Counter
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from .geometry import (build_bank, factorial_controls, geometry_report,
                       global_indices, uniform256_indices, unit_catalog)


def cells(box):
    r, c, h, w = box
    return {i * 32 + j for i in range(r, r + h) for j in range(c, c + w)}


def transpose_indices(indices):
    return sorted((i % 32) * 32 + i // 32 for i in indices)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.bank = build_bank()
        self.g = set(global_indices())

    def assert_budget(self, action):
        keep = action["keep_idx"]
        self.assertEqual(keep, sorted(set(keep)))
        self.assertEqual(len(keep), 256)
        self.assertTrue(all(type(i) is int and 0 <= i < 1024 for i in keep))
        self.assertTrue(self.g <= set(keep))

    def test_fixed_global_and_uniform_lattices(self):
        self.assertEqual(global_indices(), [r * 32 + c for r in range(0, 32, 4)
                                           for c in range(0, 32, 4)])
        self.assertEqual(uniform256_indices(), [r * 32 + c for r in range(0, 32, 2)
                                               for c in range(0, 32, 2)])
        self.assertEqual(len(self.g), 64)
        self.assert_budget({"keep_idx": uniform256_indices()})

    def test_catalog_has_70_equal_area_equal_extra_units(self):
        catalog = unit_catalog()
        self.assertEqual(len(catalog), 70)
        self.assertEqual(len({tuple(b) for b in catalog}), 70)
        self.assertEqual(Counter(tuple(b[2:]) for b in catalog), {(8, 13): 35, (13, 8): 35})
        for box in catalog:
            r, c, h, w = box
            self.assertEqual((r % 4, c % 4), (0, 0))
            self.assertTrue(r >= 0 and c >= 0 and r + h <= 32 and c + w <= 32)
            self.assertEqual(len(cells(box)), 104)
            self.assertEqual(len(cells(box) & self.g), 8)
            self.assertEqual(len(cells(box) - self.g), 96)

    def test_bank_is_12_plus_12_globally_distinct_exact_masks(self):
        self.assertEqual(Counter(a["family"] for a in self.bank), {"single": 12, "dual": 12})
        self.assertEqual(len({a["action_id"] for a in self.bank}), 24)
        self.assertEqual(len({tuple(a["keep_idx"]) for a in self.bank}), 24)
        for action in self.bank:
            self.assert_budget(action)
            a, b = cells(action["unit_a"]), cells(action["unit_b"])
            self.assertFalse(a & b)
            self.assertEqual(set(action["keep_idx"]), self.g | a | b)
            self.assertEqual(action["coverage"]["detail_raw_area"], 208)
            self.assertEqual(action["coverage"]["detail_extra_tokens"], 192)
            self.assertEqual(action["coverage"]["detail_density"], 1.0)

    def test_exact_single_adjacency_dual_permutation_and_gap(self):
        for action in self.bank:
            ri, ci = divmod(action["pair_index"], 2)
            c = ci * 16
            expected_a = [ri * 8, c, 8, 13]
            expected_b = ([ri * 8 + 8, c, 8, 13] if action["family"] == "single"
                          else [((ri + 1) % 3) * 8 + 8, 16 - c, 8, 13])
            if action["orientation"] == "horizontal":
                expected_a = [expected_a[i] for i in (1, 0, 3, 2)]
                expected_b = [expected_b[i] for i in (1, 0, 3, 2)]
            self.assertEqual(action["unit_a"], expected_a)
            self.assertEqual(action["unit_b"], expected_b)
            coverage = action["coverage"]
            if action["family"] == "single":
                self.assertEqual(coverage["detail_connected_components"], 1)
                self.assertEqual(coverage["bounding_box_area"], 208)
            else:
                self.assertEqual(coverage["detail_connected_components"], 2)
                self.assertGreaterEqual(max(coverage["row_gap"], coverage["column_gap"]), 3)

    def test_A_B_marginals_and_every_token_frequency_match(self):
        pools = [[a for a in self.bank if a["family"] == f] for f in ("single", "dual")]
        for role in ("unit_a", "unit_b"):
            self.assertEqual(*(Counter(tuple(a[role]) for a in p) for p in pools))
        counts = [Counter(i for a in p for i in a["keep_idx"]) for p in pools]
        self.assertEqual(*counts)

    def test_bank_and_factorial_transpose_symmetry(self):
        lookup = {a["action_id"]: a for a in self.bank}
        for vertical in (a for a in self.bank if a["orientation"] == "vertical"):
            horizontal = lookup[vertical["action_id"].replace("vertical", "horizontal")]
            self.assertEqual(transpose_indices(vertical["keep_idx"]), horizontal["keep_idx"])
            if vertical["family"] == "dual":
                for v, h in zip(factorial_controls(vertical), factorial_controls(horizontal)):
                    self.assertEqual((v["condition"], v["replicate"]), (h["condition"], h["replicate"]))
                    self.assertEqual(transpose_indices(v["keep_idx"]), h["keep_idx"])

    def test_factorial_all_actions_exact_replacements_disjoint_donors(self):
        catalog = {tuple(b) for b in unit_catalog()}
        for dual in (a for a in self.bank if a["family"] == "dual"):
            controls = factorial_controls(dual)
            self.assertEqual(len(controls), 6)
            self.assertEqual(len({tuple(a["keep_idx"]) for a in controls} | {tuple(dual["keep_idx"])}), 7)
            self.assertEqual({a["replicate"] for a in controls}, {0, 1})
            self.assertEqual(Counter(a["condition"] for a in controls), {"00": 2, "10": 2, "01": 2})
            donor_pairs = set()
            for control in controls:
                self.assert_budget(control)
                self.assertEqual(control["unit_a"], dual["unit_a"])
                self.assertEqual(control["unit_b"], dual["unit_b"])
                a, b = cells(control["unit_a"]), cells(control["unit_b"])
                fa, fb = cells(control["unit_FA"]), cells(control["unit_FB"])
                for name in ("unit_FA", "unit_FB"):
                    self.assertIn(tuple(control[name]), catalog)
                self.assertFalse(fa & (a | b | fb))
                self.assertFalse(fb & (a | b))
                expected = {"00": self.g | fa | fb, "10": self.g | a | fb, "01": self.g | fa | b}
                self.assertEqual(set(control["keep_idx"]), expected[control["condition"]])
                self.assertEqual(control["coverage"]["detail_raw_area"], 208)
                donor_pairs.add((tuple(control["unit_FA"]), tuple(control["unit_FB"])))
            self.assertEqual(len(donor_pairs), 2)
            self.assertEqual(factorial_controls(dual, replicates=1), controls[:3])

    def test_determinism_json_roundtrip_and_no_shared_mutable_state(self):
        original = json.dumps(self.bank, sort_keys=True)
        self.bank[0]["keep_idx"].clear()
        self.assertEqual(json.dumps(build_bank(), sort_keys=True), original)
        dual = next(a for a in build_bank() if a["family"] == "dual")
        controls = factorial_controls(dual)
        self.assertEqual(factorial_controls(json.loads(json.dumps(dual))), controls)
        controls[0]["keep_idx"].clear()
        self.assertEqual(len(factorial_controls(dual)[0]["keep_idx"]), 256)

    def test_reject_invalid_actions_replicates_and_insufficient_space(self):
        dual = next(a for a in self.bank if a["family"] == "dual")
        for replicates in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                factorial_controls(dual, replicates)
        for bad in (self.bank[0], None, {"action_id": "invented"}):
            with self.assertRaises(ValueError):
                factorial_controls(bad)
        corrupt = deepcopy(dual)
        corrupt["keep_idx"][0] = 999
        with self.assertRaises(ValueError):
            factorial_controls(corrupt)
        with patch(f"{factorial_controls.__module__}.unit_catalog", return_value=[]):
            with self.assertRaisesRegex(ValueError, "supplies only 0"):
                factorial_controls(dual)

    def test_geometry_report_invariants(self):
        report = geometry_report()
        json.dumps(report)
        self.assertEqual(report["unique_bank_masks"], 24)
        self.assertEqual(report["unit_marginals_match"], {"unit_a": True, "unit_b": True})
        self.assertTrue(report["per_token_selection_frequency_matches"])
        self.assertEqual(set(report["factorial_control_counts"].values()), {6})
        for pool in report["pools"].values():
            self.assertEqual(pool["count"], 12)
            self.assertEqual(pool["orientation_counts"], {"vertical": 6, "horizontal": 6})
            self.assertEqual(pool["detail_raw_areas"], [208])
            self.assertEqual(pool["detail_extra_tokens"], [192])


if __name__ == "__main__":
    unittest.main()
