import os
import time
import unittest
import worker


class Tests(unittest.TestCase):
    def test_actual_process_clock_identity(self):
        info = worker.clock_info()
        self.assertEqual(info['time_namespace'], os.readlink('/proc/self/ns/time'))
        self.assertRegex(info['time_namespace'], r'^time:\[\d+\]$')
        self.assertEqual(info['monotonic_clock'], time.get_clock_info('monotonic').implementation)
        self.assertEqual(info['perf_counter_clock'], time.get_clock_info('perf_counter').implementation)
    def test_entry_is_locked(self):
        with self.assertRaises(RuntimeError): worker.main()


if __name__ == '__main__': unittest.main()
