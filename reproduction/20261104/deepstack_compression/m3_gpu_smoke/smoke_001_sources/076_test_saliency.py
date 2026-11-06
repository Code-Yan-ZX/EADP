import unittest
import torch
from saliency import utilities

class SaliencyTests(unittest.TestCase):
    def test_four_streams_mean_absolute_product_in_float32(self):
        base = torch.arange(1024, dtype=torch.float32).reshape(-1, 1)
        features = [torch.cat((base, -base), 1).to(torch.bfloat16) for _ in range(4)]
        grads = [torch.tensor([1., 3.]).expand(1024, -1).to(torch.bfloat16) for _ in range(4)]
        result = utilities(features, grads)
        for scores in result:
            self.assertEqual(len(scores), 1024)
            self.assertEqual(scores[128], 256.)
            self.assertEqual(scores[0], 0.)

    def test_invalid_or_nonfinite_stream_rejected(self):
        z = torch.ones(1024, 2)
        with self.assertRaises(ValueError): utilities([z]*3, [z]*4)
        with self.assertRaises(ValueError): utilities([z]*4, [z]*3+[z[:-1]])
        bad = z.clone(); bad[0, 0] = float('nan')
        with self.assertRaises(ValueError): utilities([bad]+[z]*3, [z]*4)

if __name__ == '__main__':
    unittest.main()
