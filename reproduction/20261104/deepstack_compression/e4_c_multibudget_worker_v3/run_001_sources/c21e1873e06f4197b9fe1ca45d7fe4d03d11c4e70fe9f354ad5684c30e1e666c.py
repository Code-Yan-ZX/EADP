import importlib.util,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_multi_timing_admission_test',HERE/'admission.py');admission=importlib.util.module_from_spec(spec);spec.loader.exec_module(admission)
spec=importlib.util.spec_from_file_location('_multi_timing_worker_test',HERE/'worker.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
spec=importlib.util.spec_from_file_location('_multi_timing_launch_test',HERE/'launch.py');launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
spec=importlib.util.spec_from_file_location('_multi_fixed_reference_test',HERE/'fixed_reference_gate.py');reference_gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference_gate)
class Tests(unittest.TestCase):
 def test_gpu_and_cpu_container_stay_locked(self):
  with patch.object(admission,'plan',return_value={}),patch.object(admission,'admit_cpu_preflight'),patch.object(Path,'is_file',return_value=False):
   with self.assertRaisesRegex(ValueError,'GPU locked'):admission.admit_gpu()
  cpu={'status':'passed','test_returncode':0,'predictions_opened':True,'model_loaded':False,'GPU_authorized':False,'plan_sha256':'digest','source_sha256':{}}
  start={'cwd':str(HERE),'command':['python','freeze_execution.py']}
  with tempfile.TemporaryDirectory() as d,patch.object(admission,'REV',Path(d)),patch.object(admission,'plan',return_value={'source_sha256':{}}),patch.object(admission,'read',return_value=cpu),patch.object(admission,'terminal',return_value=start),patch.object(admission,'sha',return_value='digest'):
   with self.assertRaisesRegex(ValueError,'CPU container locked'):admission.admit_cpu_preflight()
 def test_no_gpu_worker_admission_before_container_or_processor(self):
  with tempfile.TemporaryDirectory() as d,patch.object(sys,'argv',['worker.py','--out',d]),patch.object(worker.admission,'admit_gpu',side_effect=ValueError('not admitted')),patch.object(worker,'verify_container',side_effect=AssertionError('must not start')):
   with self.assertRaisesRegex(ValueError,'not admitted'):worker.main()
 def test_preloaded_foreign_module_is_rejected(self):
  fake=types.SimpleNamespace(__file__='/tmp/foreign.py')
  with patch.dict(sys.modules,{'_multi_budget_conflict':fake}):
   with self.assertRaisesRegex(ValueError,'foreign module'):worker.load('_multi_budget_conflict',HERE/'runner.py')
 def test_partial_identity_bounded_retry(self):
  times=iter([0,0,1]);partial=json.JSONDecodeError('partial','{',1)
  with patch.object(worker.admission,'read',side_effect=[partial,{'container_id':'fixture'}]):
   self.assertEqual(worker.read_identity('p',timeout=2,now=lambda:next(times),sleep=lambda _n:None),{'container_id':'fixture'})
  times=iter([0,0,2])
  with patch.object(worker.admission,'read',side_effect=partial):
   with self.assertRaisesRegex(ValueError,'not complete'):worker.read_identity('p',timeout=2,now=lambda:next(times),sleep=lambda _n:None)
 def test_supervisor_started_bounded_retry(self):
  times=iter([0,0,1]);partial=json.JSONDecodeError('partial','{',1)
  with patch.object(launch.admission,'read',side_effect=[partial,{'worker_pid':12}]):
   self.assertEqual(launch.read_started('p',timeout=2,now=lambda:next(times),sleep=lambda _n:None),{'worker_pid':12})
  times=iter([0,0,2])
  with patch.object(launch.admission,'read',side_effect=FileNotFoundError()):
   with self.assertRaisesRegex(ValueError,'not complete'):launch.read_started('p',timeout=2,now=lambda:next(times),sleep=lambda _n:None)
 def test_missing_cpu_review_cannot_create_container(self):
  with patch.object(sys,'argv',['launch.py','--run-id','001','--preflight-only']),patch.object(launch.admission,'plan',return_value={}),patch.object(launch.admission,'admit_cpu_preflight',side_effect=ValueError('review missing')),patch.object(Path,'mkdir',side_effect=AssertionError('must not create')):
   with self.assertRaisesRegex(ValueError,'review missing'):launch.main()
 def test_timing_supervisor_scope(self):
  p={'outer_timeout_seconds':7320,'stale_seconds':1800};pid=12;cmd=['python',str(HERE/'launch.py'),'--run-id','001'];parent=['python',str(HERE.parent/'e4_c_latency/timing_supervise.py')]
  s={'worker_pid':pid,'process_group':pid,'command':cmd,'cwd':str(HERE),'timeout_seconds':7320,'stale_seconds':1800,'progress_paths':[str(HERE/'run_001/journal.jsonl')],'snapshot_seconds':3600,'poll_seconds':5,'grace_seconds':90}
  launch.supervisor(s,'001',False,parent,pid,pid,p)
  for key,value in [('timeout_seconds',1),('stale_seconds',1),('progress_paths',[]),('grace_seconds',1)]:
   with self.assertRaisesRegex(ValueError,'supervisor'):launch.supervisor(dict(s,**{key:value}),'001',False,parent,pid,pid,p)
 def test_reference_formal_evidence_tamper_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   paths={name:Path(d)/name for name in ('journal_sha256','launch_sha256','exit_sha256','worker_complete_sha256','verifier_sha256','verification_plan_sha256')}
   for p in paths.values():p.write_text('fixture')
   proof={name:reference_gate.sha(p) for name,p in paths.items()};self.assertTrue(reference_gate.verify_evidence_hashes(proof,paths))
   for name,p in paths.items():
    p.write_text('changed')
    with self.assertRaisesRegex(ValueError,'formal evidence changed'):reference_gate.verify_evidence_hashes(proof,paths)
    p.write_text('fixture')
if __name__=='__main__':unittest.main()
