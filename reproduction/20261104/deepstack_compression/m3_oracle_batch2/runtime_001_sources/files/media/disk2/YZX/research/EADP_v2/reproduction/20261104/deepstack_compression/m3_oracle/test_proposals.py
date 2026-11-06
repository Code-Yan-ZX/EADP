import unittest
from unittest.mock import patch
from proposals import N, proposals, rank, replacement


class ProposalTests(unittest.TestCase):
    def setUp(self):
        self.selected = list(range(255, -1, -1))
        self.streams = [[float(i) for i in range(N)] for _ in range(4)]

    def test_rank_tie_breaks_in_favor_of_smaller_id(self):
        r = rank([1.] * N)
        self.assertEqual(r[0], 1.)
        self.assertEqual(r[-1], 0.)
        self.assertTrue(all(r[i] > r[i+1] for i in range(N-1)))
        with self.assertRaises(ValueError): rank([float('nan')] * N)

    def test_fixed_proposals_and_equal_search_budget(self):
        result = proposals('q42', self.selected, self.streams)
        self.assertEqual(result, proposals('q42', self.selected, self.streams))
        self.assertEqual(result['pools']['MAIN'], list(range(1023, 959, -1)))
        self.assertEqual(result['pools']['MULTI'], result['pools']['MAIN'])
        for r in (8, 16):
            section = result['by_r'][str(r)]
            self.assertEqual(section['removed'], self.selected[-r:])
            self.assertEqual(section['n_proposals_per_source'], 8)
            for source, sets in section['sources'].items():
                self.assertEqual(len(sets), len({tuple(s) for s in sets}))
                self.assertEqual(sets[0], sorted(result['pools'][source][:r]))
                for s in sets:
                    self.assertEqual(len(s), r)
                    self.assertFalse(set(s) & set(self.selected))
                    new = replacement(self.selected, s, r)
                    self.assertEqual(len(new), 256)
                    self.assertEqual(set(new), set(self.selected[:-r]) | set(s))
                    self.assertEqual(new, sorted(new))

    def test_input_identity_and_fail_closed(self):
        a = proposals('q42', self.selected, self.streams)
        b = proposals('q43', self.selected, self.streams)
        self.assertNotEqual(a['pools']['RANDOM'], b['pools']['RANDOM'])
        for selected in (self.selected[:-1], [0]*256, self.selected[:-1]+[1024]):
            with self.assertRaises(ValueError): proposals('q42', selected, self.streams)
        with self.assertRaises(ValueError): proposals('q42', self.selected, self.streams[:3])
        with self.assertRaises(ValueError): replacement(self.selected, self.selected[:8], 8)
        with self.assertRaises(ValueError): replacement(self.selected[:-1] + [1024], list(range(256, 264)), 8)

    def test_exhausted_draws_reduce_all_sources_equally(self):
        class Repeating:
            def __init__(self, seed): pass
            def sample(self, population, count): return list(population)[:count]
        with patch('proposals.random.Random', Repeating):
            result = proposals('q42', self.selected, self.streams)
        for r in (8, 16):
            section = result['by_r'][str(r)]
            self.assertEqual(section['n_proposals_per_source'], 1)
            self.assertEqual([len(v) for v in section['sources'].values()], [1, 1, 1])

if __name__ == '__main__':
    unittest.main()
