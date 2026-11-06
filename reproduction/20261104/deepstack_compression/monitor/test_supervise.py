#!/usr/bin/env python3
"""CPU-only integration tests; all workers are disposable local subprocesses."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SUPERVISOR = Path(__file__).with_name('supervise.py')


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='eadp_monitor_test_')
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_monitor(self, worker, *options):
        run = self.base / 'run'
        argv = [sys.executable, str(SUPERVISOR), '--run-dir', str(run), '--timeout', '3',
                '--snapshot-seconds', '.2', '--poll-seconds', '.05', '--grace-seconds', '.2',
                *options, '--', sys.executable, '-c', worker]
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return completed, json.loads((run / 'terminal.json').read_text()), run

    def test_completed_and_hourly_equivalent_snapshots(self):
        cp, terminal, run = self.run_monitor('import time; time.sleep(.7); print("done")')
        self.assertEqual(cp.returncode, 0, cp.stderr)
        self.assertEqual(terminal['reason'], 'completed')
        self.assertEqual(terminal['remaining_process_group'], [])
        rows = [json.loads(x) for x in (run / 'health.jsonl').read_text().splitlines()]
        self.assertGreaterEqual(len(rows), 3)
        self.assertIn('gpu', rows[0])
        self.assertIn('memory', rows[0])
        self.assertIn('disk', rows[0])
        self.assertIn('done', (run / 'stdout.log').read_text())

    def test_timeout_does_not_kill_unrelated_process(self):
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(15)'])
        try:
            cp, terminal, _ = self.run_monitor('import time; time.sleep(15)')
            self.assertNotEqual(cp.returncode, 0)
            self.assertEqual(terminal['reason'], 'hard_timeout')
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=3)

    def test_stale_file_and_fresh_progress(self):
        progress = self.base / 'progress.txt'
        cp, terminal, _ = self.run_monitor('import time; time.sleep(10)', '--stale', '.5', '--progress', str(progress))
        self.assertNotEqual(cp.returncode, 0)
        self.assertEqual(terminal['reason'], 'progress_stale')

    def test_progress_prevents_false_stale(self):
        progress = self.base / 'progress.txt'
        worker = ('import time\nfrom pathlib import Path\np=Path(%r)\n'
                  'for i in range(8):\n p.write_text(str(i)); time.sleep(.15)\n' % str(progress))
        cp, terminal, _ = self.run_monitor(worker, '--stale', '.5', '--progress', str(progress))
        self.assertEqual(cp.returncode, 0, cp.stderr)
        self.assertEqual(terminal['reason'], 'completed')

    def test_existing_run_dir_is_never_reused(self):
        run = self.base / 'run'
        run.mkdir()
        (run / 'precious').write_text('unchanged')
        cp = subprocess.run([sys.executable, str(SUPERVISOR), '--run-dir', str(run),
                             '--timeout', '1', '--', sys.executable, '-c', 'pass'],
                            capture_output=True, text=True, timeout=5)
        self.assertNotEqual(cp.returncode, 0)
        self.assertEqual((run / 'precious').read_text(), 'unchanged')


if __name__ == '__main__':
    unittest.main()
