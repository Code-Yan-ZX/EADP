import math
import unittest

import torch

from rope_bias import distance_log_bias


class RopeBiasTest(unittest.TestCase):
    def test_axis_weighting_and_equal_positions(self):
        # Two temporal frequencies, one height and one width: explicit average.
        pos = torch.tensor([[0, 1], [0, 2], [0, 3]])
        freq = torch.tensor([1., .5, 1., .5])
        axes = torch.tensor([0, 0, 1, 2])
        actual = distance_log_bias(pos, pos, freq, axes)
        expected_decay = (math.cos(1.) + math.cos(.5) + math.cos(2.) + math.cos(1.5)) / 4
        torch.testing.assert_close(actual.diag(), torch.zeros(2))
        self.assertAlmostEqual(actual[1, 0].item(), math.log(2. - expected_decay), places=6)
        torch.testing.assert_close(actual, actual.T)

    def test_decode_positions_and_bad_axis(self):
        query = torch.tensor([[3], [3], [3]])
        keys = torch.zeros(3, 4, dtype=torch.long)
        bias = distance_log_bias(query, keys, torch.tensor([1.]), torch.tensor([0]))
        self.assertEqual(tuple(bias.shape), (1, 4))
        with self.assertRaises(ValueError):
            distance_log_bias(query, keys, torch.tensor([1.]), torch.tensor([3]))


if __name__ == "__main__":
    unittest.main()
