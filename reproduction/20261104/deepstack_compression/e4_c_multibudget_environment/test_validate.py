import copy
import unittest
from validate import validate, overlaps


class TestBackground(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(event='snapshot', index=0, time_namespace='time:[1]', query_start_ns=10,
                          query_end_ns=20, query_start_utc='a', query_end_utc='b', returncode=0, stdout='ps'),
                     dict(event='logging_complete', index=0, time_namespace='time:[1]', query_start_ns=10,
                          first_fsync_end_ns=30, first_fsync_end_utc='c'),
                     dict(event='watch_end', time_namespace='time:[1]', end_ns=40, end_utc='d', reason='stop_marker', samples=1)]

    def test_query_and_fsync(self):
        w = validate(self.rows, 'time:[1]')
        self.assertEqual(overlaps(w, 15, 16)['query_indices'], [0])
        x = overlaps(w, 25, 26)
        self.assertEqual(x['query_indices'], [])
        self.assertEqual(x['known_query_and_first_fsync_indices'], [0])
        self.assertEqual(x['second_fsync_overlap'], 'unknown')

    def test_invalid_inputs(self):
        for rows, ns in [(self.rows[:-1], 'time:[1]'), (self.rows, 'time:[2]')]:
            with self.assertRaises(ValueError): validate(rows, ns)
        for change in [('returncode', 1), ('query_end_ns', 9), ('index', 1)]:
            rows = copy.deepcopy(self.rows); rows[0][change[0]] = change[1]
            with self.assertRaises(ValueError): validate(rows, 'time:[1]')
        rows = copy.deepcopy(self.rows); rows[-1]['reason'] = 'duration_expired'
        with self.assertRaises(ValueError): validate(rows, 'time:[1]')
        with self.assertRaises(ValueError): overlaps([(1, 2, 3)], 3, 3)


if __name__ == '__main__':
    unittest.main()
