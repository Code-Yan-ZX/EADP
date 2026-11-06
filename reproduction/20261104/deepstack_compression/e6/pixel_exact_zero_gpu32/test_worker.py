import unittest
import torch
from worker import imports


class TestSelector(unittest.TestCase):
    def test_exact_zero_boundaries(self):
        _, author, _, candidate = imports()
        for dtype in (torch.float32, torch.bfloat16):
            tiny = torch.finfo(dtype).tiny / 2
            values = torch.tensor([0., -0., tiny, -tiny, 1., 1., 0., 1.,
                                   0., 0., 1., 0.], dtype=dtype).reshape(12, 1)
            for h, w in ((1, 12), (12, 1), (3, 4), (4, 3)):
                expected = author(method='max', threshold=0., anchored=True)._select_2d_loco_anchored(values, h, w, values.device)
                self.assertTrue(torch.equal(expected, candidate(values, h, w)))

    def test_invalid_shape(self):
        _, _, _, candidate = imports()
        for args in ((torch.zeros((4, 1)), 2, 3), (torch.zeros(4), 2, 2),
                     (torch.zeros((0, 1)), 0, 1)):
            with self.assertRaises(ValueError):
                candidate(*args)

    def test_no_cuda(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main(verbosity=2)
