"""CPU-only control-plane checks; no new model outputs or running-prefix scoring."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import supervise_after_pope_alpha as waiter


class GateTests(unittest.TestCase):
    def test_default_check_never_registers_or_launches(self):
        output = io.StringIO()
        with (mock.patch.object(waiter, 'save', side_effect=AssertionError('check attempted mutation')),
             mock.patch.object(waiter.subprocess, 'Popen', side_effect=AssertionError('check attempted model launch')),
             contextlib.redirect_stdout(output)):
            waiter.main(False, None)
        result = json.loads(output.getvalue())
        self.assertFalse(result['would_launch'])
        self.assertEqual(result['gate']['predecessor_status'], 'prepared_not_scheduled')

    def test_predecessor_failure_file_stops_even_if_plain_state_was_prepared(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'supervisor_failure.json').write_text('{}')
            with mock.patch.object(waiter, 'POPE', root), self.assertRaises(RuntimeError):
                waiter.predecessor_gate()

    def test_predecessor_supervisor_terminal_failure_and_nonzero_exit_stop(self):
        for record in ({'status': 'terminal_failure_no_retry'}, {'status': 'complete', 'exit_code': 3}):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / 'supervisor_state.json').write_text(json.dumps(record))
                with mock.patch.object(waiter, 'POPE', root), self.assertRaises((RuntimeError, AssertionError)):
                    waiter.predecessor_gate()

    def test_original_supervisor_disappearance_stops_without_timeout(self):
        record = {'status': 'waiting_NeXT_full5000', 'identity': {'pid': 91, 'startticks': 2}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'supervisor_state.json').write_text(json.dumps(record))
            with mock.patch.object(waiter, 'POPE', root), mock.patch.object(waiter, 'process_identity', return_value=None), self.assertRaises(RuntimeError):
                waiter.predecessor_liveness({'status': 'prepared_not_scheduled'}, record)

    def fixture_generating(self, root):
        (root / 'process_identity.json').write_text(json.dumps({'worker_pid': 92, 'worker_start_ticks': 3, 'controller_pid': 93}))
        (root / 'state.json').write_text(json.dumps({'status': 'generating'}))
        return {'controller': {'pid': 93, 'startticks': 4}}

    def test_worker_and_controller_disappearance_stops_generating_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            supervisor = self.fixture_generating(root)
            with mock.patch.object(waiter, 'POPE', root), mock.patch.object(waiter, 'process_identity', return_value=None), self.assertRaises(RuntimeError):
                waiter.predecessor_liveness({'status': 'generating'}, supervisor)

    def test_worker_wait_transition_with_live_controller_is_pending(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            supervisor = self.fixture_generating(root)
            def identity(pid):
                return None if pid == 92 else {'pid': 93, 'start_ticks': 4}
            with mock.patch.object(waiter, 'POPE', root), mock.patch.object(waiter, 'process_identity', side_effect=identity):
                waiter.predecessor_liveness({'status': 'generating'}, supervisor)

    def test_missing_worker_and_controller_stop_even_without_supervisor_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture_generating(root)
            with mock.patch.object(waiter, 'POPE', root), mock.patch.object(waiter, 'process_identity', return_value=None), self.assertRaises(RuntimeError):
                waiter.predecessor_liveness({'status': 'generating'}, None)

    def test_transient_JSON_write_is_pending_and_does_not_mutate(self):
        output = io.StringIO()
        with (mock.patch.object(waiter, 'predecessor_gate', side_effect=json.JSONDecodeError('transient', '', 0)),
             mock.patch.object(waiter, 'save', side_effect=AssertionError('check attempted mutation')),
             contextlib.redirect_stdout(output)):
            waiter.main(False, None)
        self.assertEqual(json.loads(output.getvalue())['gate']['reason'], 'waiting_atomic_predecessor_write')

    def test_exclusive_claim_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'claim.json'
            waiter.exclusive_record(path, {'attempt': 1})
            with self.assertRaises(FileExistsError):
                waiter.exclusive_record(path, {'attempt': 2})
            self.assertEqual(json.loads(path.read_text()), {'attempt': 1})

    def test_no_score_direction_or_observation_timeout_in_status_acceptance(self):
        self.assertTrue(waiter.terminal_status('complete'))
        self.assertFalse(waiter.terminal_status('prepared_not_scheduled'))
        for status in ('generation_failed', 'verification_failed', 'terminal_failure_no_retry'):
            with self.assertRaises(RuntimeError):
                waiter.terminal_status(status)

    def test_independent_score_helper_on_preserved_complete_run02_only(self):
        root = waiter.POPE.parent / 'run02_source_wait_only_k32'
        plan = waiter.read(root / 'run_plan.json')
        summary = waiter.read(root / 'full_result_summary.json')
        # Adapt only a metadata key absent in run02; no old files are written.
        summary['score_log_sha256'] = waiter.sha(root / 'official_score.log')
        with mock.patch.object(waiter, 'POPE', root):
            checked = waiter.independent_pope(plan, summary)
        self.assertEqual(checked['n_predictions'], 8910)
        self.assertAlmostEqual(checked['official_macro_F1_percent'], checked['independent_macro_F1_percent'], places=10)


if __name__ == '__main__':
    unittest.main()
