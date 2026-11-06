import unittest

from paired import analyze


def row(qid, group, eadp, seeds):
    return {"qid": qid, "group_id": group, "scores": {"eadp": eadp, **dict(zip((20261104, 20261105, 20261106), seeds))}}


class PairedTests(unittest.TestCase):
    def test_constant_differences_and_three_seed_mean(self):
        result = analyze([row("1", "a", 0.2, [0.4, 0.5, 0.6]), row("2", "b", 0.3, [0.5, 0.6, 0.7])], 1000)
        self.assertAlmostEqual(result["arm_mean_pct"]["random_three_seed_mean"], 55)
        self.assertAlmostEqual(result["contrasts_vs_eadp"]["random_three_seed_mean"]["difference_pp"], 30)
        for x in result["contrasts_vs_eadp"]["random_three_seed_mean"]["ci95_pp"]:
            self.assertAlmostEqual(x, 30)

    def test_question_weighted_group_counts(self):
        result = analyze([row("1", "a", 0, [0, 0, 0]), row("2", "a", 0, [1, 1, 1]), row("3", "b", 0, [1, 1, 1])], 100)
        self.assertEqual(result["image_groups"], 2)
        self.assertAlmostEqual(result["arm_mean_pct"]["random_three_seed_mean"], 200 / 3)

    def test_reproducible_and_order_independent_groups(self):
        rows = [row("1", "b", 0.1, [0.2, 0.3, 0.4]), row("2", "a", 0.2, [0.3, 0.4, 0.5])]
        self.assertEqual(analyze(rows, 50), analyze(rows[::-1], 50))

    def test_reject_missing_duplicate_and_invalid(self):
        good = row("1", "a", 0.1, [0.2, 0.3, 0.4])
        with self.assertRaises(ValueError): analyze([good, good], 10)
        with self.assertRaises(ValueError): analyze([row("1", "", 0, [0, 0, 0])], 10)
        with self.assertRaises(ValueError): analyze([row("1", "a", 0, [0, 0, float("nan")])], 10)
        with self.assertRaises(ValueError): analyze([{"qid": "1", "group_id": "a", "scores": {"eadp": 0}}], 10)


if __name__ == "__main__":
    unittest.main()
