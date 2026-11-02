"""Synthetic, CPU-only controller launch guard and failure-preservation checks."""
import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import controller

class FakeWorker:
    pid=987654321
    def __init__(self,*args,**kwargs):self.returncode=None
    def wait(self,timeout=None):self.returncode=1;return 1
    def poll(self):return self.returncode

class Tests(unittest.TestCase):
    def test_invalid_allocation_never_queries_or_spawns(self):
        with patch.object(controller.c,'read',side_effect=[{'run_id':'test'}, {'authorization':'wrong'}]), \
             patch.object(controller.c,'check'),patch.object(controller,'gpu_pids',side_effect=AssertionError('must not query GPU')), \
             patch.object(controller.subprocess,'Popen',side_effect=AssertionError('must not spawn')):
            with self.assertRaises(AssertionError):controller.main(controller.c.HERE/'root_gpu_allocation.json')

    def test_failed_worker_is_recorded_and_never_verified(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan=root/'plan.json';plan.write_text('{}');note_path=root/'root_gpu_allocation.json';note_path.write_text('{}')
            out=root/'execution';pre=root/'cpu_preflight.json';pre.write_text('{}')
            boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            note=dict(authorization='explicit_root_gpu_allocation',run_id='test',plan_sha256='digest',GPU_device=0,single_attempt=True,automatic_retry=False,boot_id=boot,cpu_preflight_sha256='digest',allowed_other_gpu_processes=[])
            pre_data=dict(success=True,plan_sha256='digest')
            def read(path):
                p=Path(path)
                if p==plan:return dict(run_id='test')
                if p==note_path:return note
                if p==pre:return pre_data
                return json.loads(p.read_text())
            count=[0]
            def identity(pid):
                if pid==os.getpid():return dict(pid=pid,start_ticks=1,boot_id=boot)
                count[0]+=1
                return dict(pid=pid,start_ticks=2,boot_id=boot) if count[0]==1 else None
            with patch.object(controller.c,'HERE',root),patch.object(controller.c,'PLAN',plan),patch.object(controller.c,'OUT',out), \
                 patch.object(controller.c,'read',side_effect=read),patch.object(controller.c,'check'),patch.object(controller.c,'sha',return_value='digest'), \
                 patch.object(controller.c,'identity',side_effect=identity),patch.object(controller,'gpu_pids',return_value=[]), \
                 patch.object(controller.subprocess,'Popen',side_effect=FakeWorker):
                with self.assertRaises(RuntimeError):controller.main(note_path)
            state=json.loads((out/'state.json').read_text())
            self.assertEqual(state['status'],'failed');self.assertEqual(state['native_exit_code'],1)
            self.assertTrue(state['original_worker_exited']);self.assertTrue(state['GPU_pid_absent'])
            self.assertFalse((out/'native_release.json').exists())

if __name__=='__main__':unittest.main()
