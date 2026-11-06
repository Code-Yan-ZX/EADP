import tempfile
import unittest
import copy
import json
import os
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import fake_sampler
import harness
import launch
import gate


class FakeQueryTest(unittest.TestCase):
    def test_only_fabricated_value_after_release_and_before_exit(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            expected = {'gpu_uuid': 'GPU-deadbeef', 'host_pid': 42}
            q = fake_sampler.fake_query(expected, d, alive=lambda _: True)
            cmd = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits']
            self.assertEqual(q(cmd).stdout, '')
            (d/'release.json').touch()
            self.assertEqual(q(cmd).stdout, 'GPU-deadbeef, 42, 1\n')
            (d/'fake_exiting').touch()
            self.assertEqual(q(cmd).stdout, '')
            with self.assertRaises(ValueError): q(['anything-else'])

    def test_dead_worker_never_appears(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td); (d/'release.json').touch()
            q = fake_sampler.fake_query({'gpu_uuid': 'GPU-deadbeef', 'host_pid': 42}, d, alive=lambda _: False)
            self.assertEqual(q(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory', '--format=csv,noheader,nounits']).stdout, '')


class CommandTest(unittest.TestCase):
    def test_supervisor_identity_and_budget_fail_closed(self):
        plan = json.loads((gate.HERE/'launch_plan.json').read_text())
        command = plan['command']; child = command[command.index('--')+1:]
        started = {'worker_pid':42, 'process_group':42, 'cwd':str(gate.BASE),
                   'timeout_seconds':360.0, 'grace_seconds':90.0,
                   'snapshot_seconds':3600.0, 'poll_seconds':2.0,
                   'stale_seconds':None, 'progress_paths':[], 'command':child}
        check = lambda value, **kw: gate.validate_supervision(
            value, plan, kw.get('pid',42), kw.get('pgid',42), kw.get('sid',42),
            kw.get('cwd',str(gate.BASE)), kw.get('parent',command), kw.get('child',child))
        check(started)
        for field, value in [('worker_pid',41), ('process_group',41),
                             ('timeout_seconds',999), ('grace_seconds',1),
                             ('command',['wrong']), ('cwd','/tmp')]:
            bad=dict(started); bad[field]=value
            with self.subTest(field=field), self.assertRaises(ValueError): check(bad)
        for kw in ({'pgid':41}, {'sid':41}, {'parent':['wrong']}, {'child':['wrong']}):
            with self.subTest(kw=kw), self.assertRaises(ValueError): check(started, **kw)

    def test_unsupervised_rejected_before_review(self):
        with patch.object(gate.time, 'monotonic', side_effect=[0, 6]):
            with self.assertRaisesRegex(ValueError, 'supervisor started'):
                gate.verify_supervision()

    def test_frozen_launch_plan_has_bounded_controller(self):
        plan = json.loads((gate.HERE/'launch_plan.json').read_text())
        cmd = plan['command']
        self.assertFalse(plan['gpu_authorized'])
        self.assertEqual(cmd[cmd.index('--timeout')+1], '360')
        self.assertEqual(cmd[cmd.index('--grace-seconds')+1], '90')
        self.assertEqual(cmd[cmd.index('--')+2:], ['e4_c_memory_handshake/launch.py', '--run-id', '001'])

    def test_actual_prior_reviewer_uses_fixed_output_path(self):
        started, terminal = gate.runtime.supervisor(gate.PRIOR_MON, gate.PRIOR_REVIEWER)
        self.assertNotIn('--out', started['command'])
        self.assertEqual(terminal['worker_returncode'], 0)
        self.assertIn("e4c_memory_runtime_container001_review.json", gate.PRIOR_REVIEWER.read_text())
    def test_private_no_device_retained_container(self):
        root = harness.BASE/'e4_c_memory_handshake'
        with tempfile.TemporaryDirectory(dir=root) as td:
            name, cmd, worker, supervisor, monitor = harness.commands(td, 'a'*32)
            self.assertEqual(name, 'eadp-memory-fake-'+'a'*32)
            self.assertNotIn('--gpus', cmd); self.assertNotIn('--device', cmd); self.assertNotIn('--rm', cmd)
            self.assertIn('CUDA_VISIBLE_DEVICES=', cmd)
            self.assertEqual(supervisor[supervisor.index('--')+1:], worker)
            self.assertEqual(monitor, Path(td).resolve()/'sampler_monitor')
            self.assertEqual(cmd[cmd.index('--network')+1], 'none')
            with self.assertRaises(ValueError): harness.commands(td, 'INVALID')
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError): harness.commands(td, 'b'*32)

    def test_locked_main(self):
        import subprocess, sys
        p = subprocess.run([sys.executable, str(Path(harness.__file__))], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('locked', p.stderr)

    def test_launch_denied_before_directory_creation(self):
        with patch.object(launch.gate, 'admit', side_effect=ValueError('review missing')):
            with self.assertRaises(ValueError): launch.launch('001')
        self.assertFalse((harness.BASE/'e4_c_memory_handshake/run_001').exists())
        with self.assertRaises(ValueError): launch.launch('999')

    def test_actual_container_mismatches_block_release(self):
        with tempfile.TemporaryDirectory(dir=harness.BASE/'e4_c_memory_handshake') as td:
            d = str(Path(td).resolve()); nonce = 'a'*32
            raw = {'Image':harness.IMAGE,
                   'HostConfig':{'NetworkMode':'none', 'DeviceRequests':None, 'Devices':[],
                                 'PidMode':'', 'CapDrop':['ALL'], 'SecurityOpt':['no-new-privileges']},
                   'Config':{'Env':['CUDA_VISIBLE_DEVICES='], 'User':f'{os.getuid()}:{os.getgid()}',
                             'Entrypoint':[harness.PYTHON],
                             'Cmd':[str(harness.BASE/'e4_c_memory_handshake/fake_worker.py'), '--directory', d, '--nonce', nonce]},
                   'Mounts':[{'Source':str(harness.PROJECT), 'Destination':str(harness.PROJECT), 'RW':False},
                             {'Source':'/home/dell/miniconda3', 'Destination':'/home/dell/miniconda3', 'RW':False},
                             {'Source':d, 'Destination':d, 'RW':True}]}
            obj = object.__new__(harness.CheckedAdapter); obj.directory=Path(d); obj.nonce=nonce
            with patch.object(harness.ControllerAdapter, 'observe_running', return_value={'verified':True}):
                obj.guard=SimpleNamespace(inspect=lambda _: {'raw':raw})
                self.assertEqual(obj.observe_running('id'), {'verified':True})
                for mutation in ('device', 'env', 'mount', 'image', 'cmd', 'network'):
                    bad=copy.deepcopy(raw)
                    if mutation=='device': bad['HostConfig']['DeviceRequests']=[{'Count':1}]
                    elif mutation=='env': bad['Config']['Env']=['CUDA_VISIBLE_DEVICES=0']
                    elif mutation=='mount': bad['Mounts'][0]['RW']=True
                    elif mutation=='image': bad['Image']='wrong'
                    elif mutation=='cmd': bad['Config']['Cmd'][0]='wrong'
                    else: bad['HostConfig']['NetworkMode']='host'
                    obj.guard=SimpleNamespace(inspect=lambda _, item=bad: {'raw':item})
                    with self.subTest(mutation=mutation), self.assertRaises(ValueError): obj.observe_running('id')


if __name__ == '__main__': unittest.main()
