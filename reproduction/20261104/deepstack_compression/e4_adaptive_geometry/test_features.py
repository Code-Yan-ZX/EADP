import unittest
from features import descriptors


def grid(h=32, w=32):
    xy = [(i // w + 7, i % w + 9) for i in range(h * w)]
    return [[[0] * len(xy)], [[x for x, y in xy]], [[y for x, y in xy]]]


class Tests(unittest.TestCase):
    def test_rectangular_grid_and_known_occupancy(self):
        p = grid(16, 64); r = descriptors(p, 0, list(range(128)))
        self.assertEqual(r["occupied8"], 8)
        self.assertAlmostEqual(r["entropy4"], 2)
        self.assertGreater(r["nearest_mean"], 0)

    def test_reject_duplicate_ids_or_coordinates(self):
        with self.assertRaises(ValueError): descriptors(grid(), 0, [0] * 128)
        p = grid(); p[1][0][1] = p[1][0][0]; p[2][0][1] = p[2][0][0]
        with self.assertRaises(ValueError): descriptors(p, 0, list(range(128)))

    def test_respects_offset_and_selected_identity(self):
        p = grid(); p = [[[3, 4] + axis[0]] for axis in p]
        self.assertEqual(descriptors(p, 2, list(range(128))), descriptors(grid(), 0, list(range(128))))


if __name__ == "__main__": unittest.main()
