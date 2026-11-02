"""CPU-only control-plane checks; never starts a watcher or scores new outputs."""
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest import mock

import supervise_terminal_audit as watcher


class TerminalAuditTests(unittest.TestCase):
    def terminal_state(self):
        return dict(controller_pid=watcher.CONTROLLER['pid'], plan_sha256=watcher.EXPECTED_PLAN,
                    status=watcher.TERMINAL_STATUS, launcher_exit_code=0, docker_client_pid=22)

    def prepared(self):
        return dict(docker_client=dict(pid=22, startticks=44))

    def test_default_check_is_read_only_and_never_starts_a_process(self):
        output = io.StringIO()
        with (mock.patch.object(watcher, 'save', side_effect=AssertionError('check attempted a write')),
              mock.patch.object(watcher.subprocess, 'Popen', side_effect=AssertionError('check attempted a process launch')),
              contextlib.redirect_stdout(output)):
            watcher.main('check', None)
        checked = json.loads(output.getvalue())
        self.assertFalse(checked['terminal_ready'])
        self.assertFalse(checked['would_start_watcher'])
        self.assertFalse(checked['would_launch_GPU'])

    def test_doc_worker_exit_alone_is_not_terminal(self):
        state = self.terminal_state(); state['status'] = 'running'
        with (mock.patch.object(watcher, 'identity', return_value=watcher.CONTROLLER),
              mock.patch.object(watcher, 'read', return_value=state),
              mock.patch.object(watcher, 'inspect_container', side_effect=AssertionError('should still wait'))):
            ready, gate = watcher.terminal_gate(self.prepared(), [])
        self.assertFalse(ready)
        self.assertEqual(gate['reason'], 'waiting_whole_run_packaged_controller_exit')

    def test_both_registered_task_argv_allowed_in_fixed_container(self):
        plan = watcher.read(watcher.PLAN)
        python = plan['command'][plan['command'].index('--entrypoint') + 1]
        entry = next(p for p in plan['source_sha256'] if p.endswith('/Qwen_vl/VLMEvalKit/run.py'))
        for task in ('DocVQA_VAL', 'AI2D_TEST'):
            command = [python, entry, '--model', plan['model_name'], '--data', task,
                       '--work-dir', str(Path(plan['environment']['OUTPUT_ROOT']) / 'eadp'), '--verbose']
            self.assertEqual(watcher.task_for_command(command, plan), task)
        self.assertTrue(watcher.in_container('0::/docker/' + watcher.CONTAINER + '\n'))
        self.assertFalse(watcher.in_container('0::/docker/foreign\n'))
        command[command.index('--data') + 1] = 'MMBench_DEV_EN'
        with self.assertRaises(AssertionError):
            watcher.task_for_command(command, plan)

    def test_controller_exit_without_final_state_stops(self):
        state = self.terminal_state(); state['status'] = 'running'
        with (mock.patch.object(watcher, 'identity', return_value=None),
              mock.patch.object(watcher, 'read', return_value=state), self.assertRaises(AssertionError)):
            watcher.terminal_gate(self.prepared(), [])

    def test_native_exit_one_is_preserved_as_terminal_not_treated_as_success(self):
        state = self.terminal_state(); state['launcher_exit_code'] = 1
        worker = dict(identity=dict(pid=33, startticks=55), task='AI2D_TEST')
        with (mock.patch.object(watcher, 'identity', return_value=None),
              mock.patch.object(watcher, 'read', return_value=state),
              mock.patch.object(watcher, 'inspect_container', return_value=dict(exists=False)),
              mock.patch.object(watcher, 'gpu_snapshot', return_value=('unrelated model', [99])),
              mock.patch.object(watcher.Path, 'read_text', return_value='0::/docker/unrelated\n'),
              mock.patch.object(watcher.audit, 'sha', return_value='terminal-state-sha')):
            ready, gate = watcher.terminal_gate(self.prepared(), [worker])
        self.assertTrue(ready)
        self.assertEqual(gate['launcher_state']['launcher_exit_code'], 1)

    def test_live_container_or_observed_worker_blocks_terminal(self):
        state = self.terminal_state()
        for container, pids in [(dict(exists=True, running=True), []), (dict(exists=False), [33])]:
            with (self.subTest(container=container, pids=pids),
                  mock.patch.object(watcher, 'identity', return_value=None),
                  mock.patch.object(watcher, 'read', return_value=state),
                  mock.patch.object(watcher, 'inspect_container', return_value=container),
                  mock.patch.object(watcher, 'gpu_snapshot', return_value=('', pids)), self.assertRaises(AssertionError)):
                watcher.terminal_gate(self.prepared(), [dict(identity=dict(pid=33, startticks=55), task='AI2D_TEST')])

    def test_transient_state_write_only_waits_while_controller_alive(self):
        with (mock.patch.object(watcher, 'identity', return_value=watcher.CONTROLLER),
              mock.patch.object(watcher, 'read', side_effect=json.JSONDecodeError('partial', '', 0))):
            self.assertFalse(watcher.terminal_gate(self.prepared(), [])[0])
        with (mock.patch.object(watcher, 'identity', return_value=None),
              mock.patch.object(watcher, 'read', side_effect=json.JSONDecodeError('partial', '', 0)),
              self.assertRaises(json.JSONDecodeError)):
            watcher.terminal_gate(self.prepared(), [])

    def test_full_coverage_and_empty_semantics_remain_distinct(self):
        canonical = [dict(index='1', question='synthetic', answer='A', A='red')]
        for value, valid, no_empty in [('', True, False), ('Failed to obtain answer: synthetic', False, True)]:
            predictions = [dict(canonical[0], prediction=value)]
            checked = watcher.audit.validate_predictions(canonical, predictions, 1)
            self.assertTrue(checked['coverage_complete'])
            self.assertEqual(checked['generation_valid'], valid)
            self.assertEqual(checked['planned_no_empty_gate_passed'], no_empty)


if __name__ == '__main__':
    unittest.main()
