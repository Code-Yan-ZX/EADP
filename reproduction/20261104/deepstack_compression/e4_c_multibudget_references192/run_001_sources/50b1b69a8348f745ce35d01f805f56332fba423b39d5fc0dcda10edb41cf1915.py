import importlib.util,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_bulk_gate_test',HERE/'gate.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
spec=importlib.util.spec_from_file_location('_bulk_worker_test',HERE/'worker.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
spec=importlib.util.spec_from_file_location('_bulk_launcher_test',HERE/'launch.py');launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
class Tests(unittest.TestCase):
 def test_gate_tampering_rejected_before_import(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'gate.py';p.write_text('raise RuntimeError("must not execute")')
   for module in (worker,launcher):
    with self.assertRaisesRegex(ValueError,'gate source changed'):module.check_gate(p)
 def test_upstream_gate_tamper_rejected_before_import(self):
  with patch.object(gate,'sha',return_value='wrong'):
   with self.assertRaisesRegex(ValueError,'prior identity gate changed'):gate.prior()
 def test_gpu_locked_before_model_or_preflight(self):
  with tempfile.TemporaryDirectory() as d,patch.object(sys,'argv',['worker.py','--out',d]),patch.object(worker.gate,'admit_gpu',side_effect=ValueError('no approval')),patch.object(worker,'preflight',side_effect=AssertionError('must not start')):
   with self.assertRaisesRegex(ValueError,'no approval'):worker.main()
 def test_preflight_main_consumes_actual_identity_result_schema(self):
  with tempfile.TemporaryDirectory() as d,patch.object(sys,'argv',['worker.py','--out',d,'--preflight-only']),patch.object(worker,'verify_container'),patch.object(worker,'event'),patch.object(worker,'preflight',return_value=({}, {'review_sha256':'verified-review'},{})),patch.object(worker.gate,'sha',return_value='digest'):
   worker.main();out=json.loads((Path(d)/'worker_preflight.json').read_text())
   self.assertEqual(out['prior_review_sha256'],'verified-review');self.assertFalse(out['cuda_initialized']);self.assertFalse(out['model_loaded'])
 def test_no_independent_fixed_identity_cannot_admit_gpu(self):
  with patch.object(gate,'plan',return_value={}),patch.object(gate,'prior'),patch.object(gate,'admit_cpu_preflight'),patch.object(gate,'admit_fixed_identity',side_effect=ValueError('fixed32 identity proof absent')):
   with self.assertRaisesRegex(ValueError,'fixed32 identity proof'):gate.admit_gpu()
 def test_actual_independent_fixed_identity_accepts(self):
  result=gate.admit_fixed_identity();self.assertEqual(len(result['formal_sha256']),64);self.assertEqual(len(result['review_sha256']),64)
 def test_missing_independent_preflight_review_rejected(self):
  with tempfile.TemporaryDirectory() as d,patch.object(gate,'REV',Path(d)),patch.object(gate,'execution_cpu',return_value={}):
   with self.assertRaisesRegex(ValueError,'CPU container locked'):gate.admit_cpu_preflight({})
 def test_partial_container_identity_retried_and_timeout_rejected(self):
  times=iter([0,0,1]);partial=json.JSONDecodeError('partial','{',1)
  with patch.object(worker.gate,'read',side_effect=[partial,{'container_id':'fixture'}]):
   self.assertEqual(worker.read_identity('p',timeout=2,now=lambda:next(times),sleep=lambda _n:None),{'container_id':'fixture'})
  times=iter([0,0,2])
  with patch.object(worker.gate,'read',side_effect=partial):
   with self.assertRaisesRegex(ValueError,'not complete'):worker.read_identity('p',timeout=2,now=lambda:next(times),sleep=lambda _n:None)
 def test_source_tampering(self):
  with tempfile.TemporaryDirectory() as d,patch.object(gate,'HERE',Path(d)):
   p=Path(d)/'s';p.write_text('original');plan={'stage':'222-input four-budget fixed32 reference generation','jobs':888,'source_sha256':{str(p):gate.sha(p)}}
   (Path(d)/'execution_plan001.json').write_text(json.dumps(plan));gate.plan();p.write_text('tampered')
   with self.assertRaisesRegex(ValueError,'source changed'):gate.plan()
 def test_supervisor_and_deadline(self):
  initial=gate.read(gate.BASE/'initial_plan.json');p={'outer_timeout_seconds':7320,'inner_timeout_seconds':7200,'stale_seconds':1200,'grace_seconds':90,'acceptance_reserve_seconds':3600,'budget_source_sha256':gate.sha(gate.BASE/'initial_plan.json')}
  s={'worker_pid':12,'process_group':12,'cwd':str(gate.BASE),'command':['python',str(HERE/'launch.py'),'--run-id','001'],'timeout_seconds':7320,'stale_seconds':1200,'progress_paths':[str(HERE/'run_001/diagnostic/journal.jsonl')],'snapshot_seconds':3600,'poll_seconds':5,'grace_seconds':90}
  with patch.object(gate.os,'getpid',return_value=12),patch.object(gate.os,'getpgrp',return_value=12):
   gate.check_supervisor(s,'001',initial['budget_start_unix']+1,p)
   with self.assertRaisesRegex(ValueError,'deadline'):gate.check_supervisor(s,'001',initial['budget_end_unix']-p['outer_timeout_seconds'],p)
   bad=dict(s,stale_seconds=1)
   with self.assertRaisesRegex(ValueError,'supervisor'):gate.check_supervisor(bad,'001',initial['budget_start_unix']+1,p)
 def test_failed_docker_cleanup_is_not_success(self):
  spec=importlib.util.spec_from_file_location('_bulk_test_boot',gate.BASE/'e6/pixel_natural_timing_pilot_v4/bootstrap.py');boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.pin_all();import timing_launch
  good={'returncode':0,'timed_out':False,'interrupted':None,'client_exited_while_container_running':False,'container_stopped_verified':True,'container_exit_code':0,'gpu_pids_after':[],'failure_type':None,'container_identity':{'container_id':'id','host_init':{'pid':1,'start_ticks':1,'boot_id':'b'}}}
  self.assertEqual(timing_launch.exit_code(good),0)
  for field,value in [('timed_out',True),('interrupted',15),('client_exited_while_container_running',True),('container_stopped_verified',False),('gpu_pids_after',[123]),('container_identity',None)]:
   with self.subTest(field=field):self.assertNotEqual(timing_launch.exit_code(dict(good,**{field:value})),0)
if __name__=='__main__':unittest.main()
