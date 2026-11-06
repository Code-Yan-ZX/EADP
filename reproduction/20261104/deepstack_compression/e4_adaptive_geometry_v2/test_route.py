import unittest
import route


class Tests(unittest.TestCase):
    def rows(self):
        features, scores = [], []
        for task in route.TASKS:
            for i in range(128):
                row = {"task": task, "qid": str(i), "group_id": f"{task}-{i:03}"}
                features.append({**row, "nearest_mean": i / 128, "occupied8": i % 64, "entropy4": i / 128})
                scores.append({**row, "arms": {f"K{k}-D1": {"score": v, "independent_score": v} for k, v in ((128, .2), (256, .5), (512, .8))}})
        return features, scores

    def test_full_fixed_budget_and_denominator(self):
        f, s = self.rows(); result, a = route.analyze(f, s)
        self.assertEqual(len(a), 384 * 6)
        for x in result.values():
            self.assertEqual(x["n"], 128)
            for value in x["rules"].values():
                self.assertEqual(value["high_k_count"], 42); self.assertEqual(value["mean_k"], 254)
                self.assertEqual(value["improved"] + value["declined"] + value["unchanged"], 128)

    def test_missing_group_and_score_disagreement_rejected(self):
        f, s = self.rows()
        with self.assertRaises(ValueError): route.analyze(f, s[:-1])
        s[0]["arms"]["K128-D1"]["independent_score"] = .1
        with self.assertRaises(ValueError): route.analyze(f, s)


if __name__ == "__main__": unittest.main()
