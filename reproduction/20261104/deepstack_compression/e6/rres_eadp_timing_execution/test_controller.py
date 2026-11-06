import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import controller

ROOT = Path(__file__).resolve().parents[2]
JOBS = json.loads((ROOT / 'e6/rres_eadp_timing_pilot_v4/schedule001.json').read_text())['processes']


class Runtime:
    def __init__(self, busy=False):
        self.busy = busy
    def gpu_pids(self):
        return [1] if self.busy else []
    def identity(self, pid):
        return {'pid': pid}
    def container_command(self, author, args, gpu, name):
        return ['docker', 'run', '--rm', '--name', name, *sum((['--device', d] for d in controller.DEVICES), []), '--env', 'CUDA_VISIBLE_DEVICES=0', *args]


class Bounded:
    def __init__(self, mismatch=None, fail=None):
        self.calls = []
        self.mismatch = mismatch
        self.fail = fail
    def run_bounded(self, command, name, timeout, gpu_pids, identity, directory, require_gpu_idle):
        n = len(self.calls)
        self.calls.append((command, name))
        if self.fail == n:
            return 1
        result = {'order': n, 'requests': len(JOBS[n]['requests']), 'all_completed': True, 'identity_mismatches': int(self.mismatch == n)}
        controller.write_once(directory / 'observations/worker_complete.json', result)
        return 0


def read(path):
    return json.loads(Path(path).read_text())


class Tests(unittest.TestCase):
    def run_dispatch(self, bounded, runtime=None, jobs=None):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        out = Path(self.tmp.name) / 'run'
        result = controller.dispatch(jobs if jobs is not None else JOBS, JOBS, runtime or Runtime(), bounded, {}, out, {'scope': 'fake'}, read)
        return result, out
    def test_full_order_and_counts(self):
        bounded = Bounded()
        result, out = self.run_dispatch(bounded)
        self.assertEqual(result, 0)
        self.assertEqual(len(bounded.calls), 12)
        self.assertEqual(read(out / 'controller_complete.json')['status'], 'passed')
        self.assertTrue(all('--rm' not in command for command, _ in bounded.calls))
    def test_identity_mismatch_does_not_drop_later_processes(self):
        bounded = Bounded(mismatch=2)
        result, out = self.run_dispatch(bounded)
        self.assertEqual((result, len(bounded.calls)), (1, 12))
        self.assertEqual(read(out / 'controller_complete.json')['identity_mismatches'], 1)
    def test_worker_error_aborts_without_complete(self):
        bounded = Bounded(fail=2)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'run'
            with self.assertRaises(RuntimeError):
                controller.dispatch(JOBS, JOBS, Runtime(), bounded, {}, out, {}, read)
            self.assertEqual(len(bounded.calls), 3)
            self.assertFalse((out / 'controller_complete.json').exists())
    def test_busy_rejected(self):
        bounded = Bounded()
        with self.assertRaises(ValueError):
            self.run_dispatch(bounded, Runtime(True))
        self.assertEqual(bounded.calls, [])
    def test_sequence_tamper_rejected_before_container(self):
        bounded = Bounded()
        with self.assertRaises(ValueError):
            self.run_dispatch(bounded, jobs=list(reversed(JOBS)))
        self.assertEqual(bounded.calls, [])
    def test_request_tamper_rejected_before_container(self):
        altered = json.loads(json.dumps(JOBS))
        altered[0]['requests'][0]['qid'] = '0'
        bounded = Bounded()
        with self.assertRaises(ValueError):
            self.run_dispatch(bounded, jobs=altered)
        self.assertEqual(bounded.calls, [])
    def test_entry_remains_locked(self):
        with self.assertRaises(RuntimeError):
            controller.main()


if __name__ == '__main__':
    unittest.main()
