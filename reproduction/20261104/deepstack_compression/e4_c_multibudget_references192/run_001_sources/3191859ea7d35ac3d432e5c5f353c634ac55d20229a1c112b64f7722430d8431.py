import importlib.util,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;BASE=HERE.parent
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
launch=load('_budget_ttft_launch_test',HERE/'launch.py');worker=load('_budget_ttft_worker_test',HERE/'worker.py')
class Tests(unittest.TestCase):
 def test_real_identity_proof_is_accepted(self):self.assertIn('review_sha256',worker.identity.admit_identity())
 def test_gpu_worker_refuses_before_preflight(self):
  with tempfile.TemporaryDirectory() as d,patch.object(sys,'argv',['worker.py','--out',d]),patch.object(worker.admission,'admit_gpu',side_effect=ValueError('not approved')),patch.object(worker,'prepare',side_effect=AssertionError('must not start')):
   with self.assertRaisesRegex(ValueError,'not approved'):worker.main()
 def test_missing_final_review_rejected(self):
  with tempfile.TemporaryDirectory() as d,patch.object(launch.admission,'REV',Path(d)),patch.object(launch.admission,'plan',return_value={}),patch.object(launch.admission.identity,'admit_identity',return_value={}):
   with self.assertRaisesRegex(ValueError,'GPU locked'):launch.admission.admit_gpu()
 def test_health_window_supervisor_identity_and_stale(self):
  s={'worker_pid':12,'process_group':12,'command':['python',str(HERE/'launch.py'),'--run-id','001'],'cwd':str(BASE),'timeout_seconds':3720,'stale_seconds':1200,
     'progress_paths':[str(HERE/'run_001/journal.jsonl')],'snapshot_seconds':3600,'poll_seconds':5,'grace_seconds':90}
  parent=['python',str(BASE/'e4_c_latency/timing_supervise.py')];launch.supervisor(s,'001',False,parent,12,12)
  with self.assertRaisesRegex(ValueError,'identity'):launch.supervisor(s,'001',False,['python','monitor/supervise.py'],12,12)
  for key in ('stale_seconds','timeout_seconds','grace_seconds'):
   bad=dict(s);bad[key]=1
   with self.assertRaises(ValueError):launch.supervisor(bad,'001',False,parent,12,12)
 def test_bounded_controller_rejects_abnormal_exit(self):
  # Other test modules imported this pilot's same-named identity gate. A real
  # launcher is a fresh process and pins the E4 gate before such imports.
  names=('identity_gate','contract','runner');saved={name:sys.modules.pop(name,None) for name in names}
  try:
   boot=load('_budget_ttft_test_boot',BASE/'e6/pixel_natural_timing_pilot_v4/bootstrap.py');boot.pin_all();import timing_launch
  finally:
   for name,module in saved.items():
    if module is not None:sys.modules[name]=module
  good={'returncode':0,'timed_out':False,'interrupted':None,'client_exited_while_container_running':False,'container_stopped_verified':True,'container_exit_code':0,'gpu_pids_after':[],
        'failure_type':None,'container_identity':{'container_id':'id','host_init':{'pid':1,'start_ticks':1,'boot_id':'b'}}}
  self.assertEqual(timing_launch.exit_code(good),0)
  for key,value in [('timed_out',True),('interrupted',15),('client_exited_while_container_running',True),('container_stopped_verified',False),('gpu_pids_after',[1]),('container_identity',None)]:
   bad=dict(good);bad[key]=value;self.assertNotEqual(timing_launch.exit_code(bad),0)
