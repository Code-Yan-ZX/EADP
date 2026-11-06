import importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import probe

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('scatter32_worker_test',HERE/'worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)

class TestWorker(unittest.TestCase):
    def test_plan_rejects_resource_and_schedule_changes(self):
        p={'stage':'author scatter graph 32-image identity','cases':list(w.CASES),'child_timeout_seconds':600,
           'allocator_fraction':.6,'k512_min_free_bytes':32*1024**3,'other_min_free_bytes':12*1024**3,'sha256':{}}
        for key,value in [('allocator_fraction',.9),('child_timeout_seconds',601),('k512_min_free_bytes',1),('cases',['k_512'])]:
            bad=dict(p);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):w.validate_plan(bad)
        with self.assertRaises(ValueError):w.validate_plan(p)
    def test_bad_schedule_rejected_before_cuda(self):
        with self.assertRaises(ValueError):probe.run_one(512,[],lambda _:None)
    def test_child_cleanup_and_progress(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'progress';w.event(path,{'event':'started'});w.event(path,{'event':'failed'})
            self.assertEqual([json.loads(x)['event'] for x in path.read_text().splitlines()],['started','failed'])
            p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
            try:w.stop_child(p);self.assertIsNotNone(p.poll())
            finally:
                if p.poll() is None:p.kill();p.wait()
    def test_actual_launcher_rejects_wrong_supervisor_before_gpu(self):
        launcher_spec=importlib.util.spec_from_file_location('scatter32_launch_test',HERE/'launch.py')
        launcher=importlib.util.module_from_spec(launcher_spec);launcher_spec.loader.exec_module(launcher)
        original=Path.read_text
        def fake_read(path,*args,**kwargs):
            if path==HERE/'plan001.json':return json.dumps({'budget_source_sha256':'x'})
            if path==w.BASE/'initial_plan.json':return json.dumps({'budget_start_unix':0,'budget_end_unix':10000000000})
            if path==HERE/'monitor_smoke001/started.json':return json.dumps({'worker_pid':-1,'timeout_seconds':2520,'grace_seconds':90})
            return original(path,*args,**kwargs)
        with patch.object(sys,'argv',['launch.py','--stage','smoke']),patch.object(Path,'read_text',fake_read),\
             patch.object(launcher,'preflight'),patch.object(launcher,'sha',return_value='x'),\
             patch.object(launcher,'validate_budget_window'),patch.object(launcher,'load') as load:
            with self.assertRaises(ValueError):launcher.main()
            load.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)
