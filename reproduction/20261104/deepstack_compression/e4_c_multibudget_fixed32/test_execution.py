import importlib.util,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('budget_execution_gate_test',HERE/'gate.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
spec=importlib.util.spec_from_file_location('budget_execution_worker_test',HERE/'worker.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)

class Tests(unittest.TestCase):
 def test_actual_prior_proof_and_command(self):
  self.assertEqual(len(gate.prior()['source_sha256']),120)
 def test_missing_gpu_review_is_fail_closed(self):
  with tempfile.TemporaryDirectory() as d,patch.object(gate,'REVIEW',Path(d)),patch.object(gate,'plan',return_value={}),patch.object(gate,'prior',return_value={}),patch.object(gate,'admit_cpu_preflight',return_value={}):
   with self.assertRaisesRegex(ValueError,'GPU locked'):gate.admit_gpu()
 def test_plan_source_tamper_is_rejected(self):
  with tempfile.TemporaryDirectory() as d,patch.object(gate,'HERE',Path(d)):
   source=Path(d)/'one';source.write_text('a');p={'stage':'4-budget fixed32 24-generation diagnostic','budgets':[128,256,512,1024],'source_sha256':{str(source):gate.sha(source)}}
   (Path(d)/'execution_plan005.json').write_text(json.dumps(p));gate.plan();source.write_text('b')
   with self.assertRaisesRegex(ValueError,'source changed'):gate.plan()
 def test_gpu_worker_checks_gate_before_preflight_or_model(self):
  with tempfile.TemporaryDirectory() as d,patch.object(sys,'argv',['worker.py','--out',d]),patch.object(worker.gate,'admit_gpu',side_effect=ValueError('no approval')),patch.object(worker,'preflight',side_effect=AssertionError('must not start')):
   with self.assertRaisesRegex(ValueError,'no approval'):worker.main()
 def test_execution_cpu_requires_actual_plan_and_supervisor(self):
  with tempfile.TemporaryDirectory() as d,patch.object(gate,'HERE',Path(d)):
   root=Path(d);(root/'execution_plan005.json').write_text('{}');(root/'monitor_execution_cpu005').mkdir();(root/'freeze_execution.py').write_text('fixture')
   p={'source_sha256':{'fixture':'digest'}};proof={'status':'passed','test_returncode':0,'model_loaded':False,'GPU_authorized':False,'predictions_opened':True,'plan_sha256':gate.sha(root/'execution_plan005.json'),'source_sha256':p['source_sha256']}
   (root/'execution_cpu005.json').write_text(json.dumps(proof));start={'worker_pid':10,'cwd':str(root),'command':['python','freeze_execution.py']};end={'worker_pid':10,'reason':'completed','worker_returncode':0,'remaining_process_group':[]}
   (root/'monitor_execution_cpu005/started.json').write_text(json.dumps(start));(root/'monitor_execution_cpu005/terminal.json').write_text(json.dumps(end));gate.execution_cpu(p)
   proof['plan_sha256']='bad';(root/'execution_cpu005.json').write_text(json.dumps(proof))
   with self.assertRaisesRegex(ValueError,'CPU proof mismatch'):gate.execution_cpu(p)
   proof['plan_sha256']=gate.sha(root/'execution_plan005.json');(root/'execution_cpu005.json').write_text(json.dumps(proof));end['worker_returncode']=1;(root/'monitor_execution_cpu005/terminal.json').write_text(json.dumps(end))
   with self.assertRaisesRegex(ValueError,'not completed'):gate.execution_cpu(p)
 def test_missing_independent_preflight_review_rejected(self):
  with tempfile.TemporaryDirectory() as d,patch.object(gate,'REVIEW',Path(d)),patch.object(gate,'execution_cpu',return_value={}):
   with self.assertRaisesRegex(ValueError,'CPU container locked'):gate.admit_cpu_preflight({})
 def test_partial_container_identity_retried_and_timeout_rejected(self):
  times=iter([0,0,1]);partial=json.JSONDecodeError('partial','{',1)
  with patch.object(worker.gate,'read',side_effect=[partial,{'container_id':'fixture'}]):
   self.assertEqual(worker.read_identity('p',timeout=2,now=lambda:next(times),sleep=lambda _n:None),{'container_id':'fixture'})
  times=iter([0,0,2])
  with patch.object(worker.gate,'read',side_effect=partial):
   with self.assertRaisesRegex(ValueError,'not complete'):worker.read_identity('p',timeout=2,now=lambda:next(times),sleep=lambda _n:None)
 def test_supervisor_rejects_stale_and_timeout_changes(self):
  p={'outer_timeout_seconds':3720,'inner_timeout_seconds':3600,'stale_seconds':1200,'grace_seconds':90,'acceptance_reserve_seconds':3600,'budget_source_sha256':gate.sha(gate.BASE/'initial_plan.json')}
  initial=gate.read(gate.BASE/'initial_plan.json');cmd=['python',str(HERE/'launch.py'),'--run-id','001']
  s={'worker_pid':12,'process_group':12,'cwd':str(gate.BASE),'command':cmd,'timeout_seconds':3720,'stale_seconds':1200,'progress_paths':[str(HERE/'run_001/diagnostic/journal.jsonl')],'snapshot_seconds':3600,'poll_seconds':5,'grace_seconds':90}
  with patch.object(gate.os,'getpid',return_value=12),patch.object(gate.os,'getpgrp',return_value=12):
   gate.check_supervisor(s,'001',initial['budget_start_unix']+1,p)
   for field in ('stale_seconds','timeout_seconds','grace_seconds'):
    bad=dict(s);bad[field]=1
    with self.assertRaisesRegex(ValueError,'supervisor'):gate.check_supervisor(bad,'001',initial['budget_start_unix']+1,p)
   with self.assertRaisesRegex(ValueError,'deadline'):gate.check_supervisor(s,'001',initial['budget_end_unix']-p['outer_timeout_seconds'],p)
 def test_launcher_exit_failure_boundaries(self):
  base=HERE.parent;spec=importlib.util.spec_from_file_location('_execution_test_boot',base/'e6/pixel_natural_timing_pilot_v4/bootstrap.py');boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.pin_all()
  import timing_launch
  good={'returncode':0,'timed_out':False,'interrupted':None,'client_exited_while_container_running':False,'container_stopped_verified':True,'container_exit_code':0,'gpu_pids_after':[],
        'failure_type':None,'container_identity':{'container_id':'id','host_init':{'pid':1,'start_ticks':1,'boot_id':'b'}}}
  self.assertEqual(timing_launch.exit_code(good),0)
  for field,value in [('timed_out',True),('interrupted',15),('client_exited_while_container_running',True),('container_stopped_verified',False),('gpu_pids_after',[123]),('container_identity',None)]:
   bad=dict(good);bad[field]=value;self.assertNotEqual(timing_launch.exit_code(bad),0)
if __name__=='__main__':unittest.main()
