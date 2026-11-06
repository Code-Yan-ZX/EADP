import itertools
import unittest
import torch
from interventions import e2_slots, e3_permutation, e3_streams


class Interventions(unittest.TestCase):
    def setUp(self):
        rows = torch.arange(256, dtype=torch.float32).reshape(-1, 1)
        self.streams = [torch.cat([rows + offset, rows*2 + offset], dim=1) for offset in (0, 1000, 2000)]

    def test_all_e2_patterns_keep_original_slot_and_dtype(self):
        original = [x.clone() for x in self.streams]
        for bits in map(''.join, itertools.product('01', repeat=3)):
            out = e2_slots(torch, self.streams, bits)
            self.assertEqual(len(out), 3)
            for i in range(3):
                expected = original[i] if bits[i] == '1' else torch.zeros_like(original[i])
                self.assertTrue(torch.equal(out[i], expected), (bits, i))
                self.assertEqual(out[i].dtype, original[i].dtype)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(self.streams, original)))

    def test_perm_fixed_sha_seed_and_common_nonfixed_row_mapping(self):
        seed, expected = e3_permutation('0'*64)
        self.assertEqual(seed, 2887301250010932645)
        self.assertEqual(expected[:12], [245, 38, 213, 18, 12, 75, 79, 192, 43, 131, 70, 73])
        self.assertEqual(sorted(expected), list(range(256)))
        self.assertTrue(all(i != j for i, j in enumerate(expected)))
        out, record = e3_streams(torch, self.streams, 'PERM', '0'*64)
        self.assertEqual(record['permutation'], expected)
        for original, actual in zip(self.streams, out):
            self.assertTrue(torch.equal(actual, original[expected]))
        self.assertEqual(e3_permutation('0'*64), (seed, expected))

    def test_mean_is_selected_row_mean_per_slot(self):
        out, meta = e3_streams(torch, self.streams, 'MEAN')
        self.assertEqual(meta['mode'], 'MEAN')
        for i, z in enumerate(out):
            expected = torch.tensor([127.5 + i*1000, 255 + i*1000])
            self.assertTrue(torch.equal(z, expected.expand_as(z)))

    def test_off_and_match_preserve_three_slot_order(self):
        off, _ = e3_streams(torch, self.streams, 'OFF')
        match, _ = e3_streams(torch, self.streams, 'MATCH')
        for i in range(3):
            self.assertEqual(torch.count_nonzero(off[i]).item(), 0)
            self.assertTrue(torch.equal(match[i], self.streams[i]))

    def test_reject_invalid_geometry_bits_and_hash(self):
        for streams in (self.streams[:2], [self.streams[0], self.streams[1][:-1], self.streams[2]],
                        [self.streams[0], self.streams[1].double(), self.streams[2]]):
            with self.assertRaises(ValueError): e2_slots(torch, streams, '111')
        for bits in ('11', '1111', '1x1', None):
            with self.assertRaises(ValueError): e2_slots(torch, self.streams, bits)
        for image_hash in ('A'*64, 'x'*64, 'a'*63, None):
            with self.assertRaises(ValueError): e3_permutation(image_hash)
        with self.assertRaises(ValueError): e3_permutation('0'*64, 1)


if __name__ == '__main__': unittest.main()
