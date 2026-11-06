import json
from pathlib import Path
import tempfile
import unittest
import health_snapshot as h


class Tests(unittest.TestCase):
    def test_capture_original_bytes_and_wait(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d); windows = base/'w'; events = base/'e'; output = base/'out'; output.mkdir()
            windows.write_bytes(b'{"event":"check"}\n'); events.write_bytes(b'{"event":"done"}\n')
            with self.assertRaises(ValueError): h.capture(windows, events, output, process_end_ns=1, clock=lambda: 50_000_000_000)
            record = h.capture(windows, events, output, process_end_ns=1, clock=lambda: 51_000_000_000)
            self.assertEqual((output/'automatic_health_windows_snapshot001.jsonl').read_bytes(), windows.read_bytes())
            self.assertEqual(record['events_bytes'], len(events.read_bytes()))
            events.write_bytes(events.read_bytes()+b'{"later":true}\n')
            self.assertNotEqual((output/'automatic_health_events_snapshot001.jsonl').read_bytes(), events.read_bytes())
    def test_partial_or_bad_record_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'w'; p.write_bytes(b'{"unfinished":')
            with self.assertRaises(ValueError): h.read_stable_prefix(p)
            p.write_bytes(b'[]\n')
            with self.assertRaises(ValueError): h.read_stable_prefix(p)
    def test_entry_locked(self):
        with self.assertRaises(RuntimeError): h.main()


if __name__ == '__main__': unittest.main()
