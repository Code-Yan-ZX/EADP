import copy,json,os,sys,tempfile,types,unittest
from pathlib import Path
from collections import Counter
from unittest.mock import patch
from transformers.generation.stopping_criteria import StoppingCriteriaList
from schedule import build
from stage_runner import execute_stage

def rows():
 return [{'task':task,'qid':str(i),'group_id':task+'-'+str(i),'split':'DEV'} for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST') for i in range(128)]

class ScheduleTests(unittest.TestCase):
 def test_actual_latency_launcher_reaches_mocked_boundary_and_checks_busy_gpu(self):
  import latency_launch
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);worker=root/'latency_worker.py';worker.write_text('worker');budgetfile=root/'budget_plan.json'
   stage={'inner_timeout_seconds':3600,'outer_timeout_seconds':3720,'journal_stale_seconds':1200,'snapshot_seconds':3600,'poll_seconds':15,'grace_seconds':90,'minimum_free_bytes':0}
   budgetfile.write_text(json.dumps({'budget_source':'../unused','smoke':stage}));src=[worker,budgetfile];approval={'source_sha256':{str(x):latency_launch.sha(x) for x in src}}
   def started(run):
    p=root/('monitor_latency'+run);p.mkdir();(p/'started.json').write_text(json.dumps({'worker_pid':os.getpid(),'timeout_seconds':3720,'stale_seconds':1200,'progress_paths':[str(root/('latency_'+run)/'journal.jsonl')],'snapshot_seconds':3600,'poll_seconds':15,'grace_seconds':90}))
   started('test');started('busy');calls=[];mod=types.ModuleType('run_panel')
   mod.container_command=lambda plan,args,gpu,name:(calls.append((args,gpu)) or ['docker','run','--rm','image',*args]);mod.gpu_pids=lambda:[];mod.identity=lambda pid:{'pid':pid}
   old=list(sys.path)
   try:
    with patch.object(latency_launch,'HERE',root),patch.object(latency_launch,'sources',return_value=src),patch.object(latency_launch,'admit',return_value=approval),patch.object(latency_launch,'validate_budget_window'),patch.object(latency_launch,'run_bounded',return_value=0),patch.dict(sys.modules,{'run_panel':mod}),patch('sys.argv',['latency_launch.py','--stage','smoke','--run-id','test']):
     self.assertEqual(latency_launch.main(),0);self.assertEqual(len(calls),1);self.assertIs(calls[0][1],True)
     mod.gpu_pids=lambda:[777]
     with patch('sys.argv',['latency_launch.py','--stage','smoke','--run-id','busy']):
      with self.assertRaisesRegex(ValueError,'GPU occupied'):latency_launch.main()
     self.assertFalse((root/'latency_busy').exists());self.assertEqual(len(calls),1)
   finally:sys.path[:]=old
 def test_actual_cpu_worker_launcher_uses_no_gpu_at_mocked_boundary(self):
  import preflight_worker
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);worker=root/'latency_worker.py';worker.write_text('worker');budget=root/'budget_plan.json';budget.write_text(json.dumps({'budget_source':'../unused'}));src=[worker,budget]
   approval={'source_sha256':{str(x):preflight_worker.sha(x) for x in src}};m=root/'monitor_worker_preflightx';m.mkdir();(m/'started.json').write_text(json.dumps({'worker_pid':os.getpid(),'timeout_seconds':1500,'grace_seconds':90}))
   calls=[];mod=types.ModuleType('run_panel');mod.container_command=lambda plan,args,gpu,name:(calls.append((args,gpu)) or ['docker','run','--rm','image',*args]);mod.identity=lambda pid:{'pid':pid}
   old=list(sys.path)
   try:
    with patch.object(preflight_worker,'HERE',root),patch.object(preflight_worker,'sources',return_value=src),patch.object(preflight_worker,'admit',return_value=approval),patch.object(preflight_worker,'validate_budget_window'),patch.object(preflight_worker,'run_bounded',return_value=0) as bounded,patch.dict(sys.modules,{'run_panel':mod}),patch('sys.argv',['preflight_worker.py','--run-id','x']):
     self.assertEqual(preflight_worker.main(),0);self.assertIs(calls[0][1],False);self.assertIsNone(bounded.call_args.args[3]);self.assertIs(bounded.call_args.kwargs['require_gpu_idle'],False)
   finally:sys.path[:]=old
 def test_cpu_probe_never_claims_gpu_idle(self):
  from timing_launch import exit_code
  host={'pid':1,'start_ticks':2,'boot_id':'boot'};r={'returncode':0,'container_exit_code':0,'timed_out':False,'interrupted':None,'failure_type':None,
    'client_exited_while_container_running':False,'container_stopped_verified':True,'container_identity':{'container_id':'x','host_init':host},'gpu_pids_after':None}
  self.assertEqual(exit_code(r),1);self.assertEqual(exit_code(r,require_gpu_idle=False),0)
 def test_missing_light_identity_cannot_admit_timing(self):
  import tempfile,latency_gate
  from pathlib import Path
  with tempfile.TemporaryDirectory() as d,patch.object(latency_gate,'C',Path(d)),patch.object(latency_gate,'REVIEW',Path(d)):
   with self.assertRaisesRegex(ValueError,'light identity proof missing'):latency_gate.require_light_identity()
 def test_smoke_is_fixed_72_real_schedule_subset(self):
  r=rows();s=build(r,[],'smoke');self.assertEqual(s,build(r,[],'smoke'));self.assertEqual(len(s['schedule']),72)
  self.assertEqual(Counter(x['phase'] for x in s['schedule']),{'warmup':36,'measure':36})
  self.assertEqual(Counter(x['protocol'] for x in s['schedule']),{'natural':36,'fixed32':36})
  self.assertEqual(Counter(x['mode'] for x in s['schedule']),{'C-ON':24,'C-DROP':24,'C-SKIP':24})
  self.assertEqual([x['source_order'] for x in s['schedule']],sorted(x['source_order'] for x in s['schedule']))
 def test_all_stage_events_retain_order_and_no_dropped_jobs(self):
  r=rows();s=build(r,[],'smoke');events=[]
  def execute(_row,mode,protocol,_partial):return {'mode':mode,'protocol':protocol}
  self.assertEqual(execute_stage(s,r,[],'smoke',execute,events.append),72)
  self.assertEqual([e['status'] for e in events],['generated','completed']*72)
 def test_journal_retains_partial_output_before_failure(self):
  r=rows();s=build(r,[],'smoke');events=[]
  def fail(_row,_mode,_protocol,partial):partial({'output_ids':[1]});raise RuntimeError('post-generate failure')
  with self.assertRaisesRegex(RuntimeError,'post-generate'):execute_stage(s,r,[],'smoke',fail,events.append)
  self.assertEqual([x['status'] for x in events],['generated_partial','failed'])
  bad=copy.deepcopy(s);bad['schedule'][0]['mode']='C-ON' if bad['schedule'][0]['mode']!='C-ON' else 'C-DROP'
  with self.assertRaisesRegex(ValueError,'frozen schedule'):execute_stage(bad,r,[],'smoke',fail,events.append)
 def test_main_worker_fails_before_model_without_identity(self):
  import latency_worker,latency_admission
  with patch('sys.argv',['latency_worker.py','--stage','smoke','--out','unused']),patch.object(latency_admission,'admit',side_effect=ValueError('identity missing')):
   with self.assertRaisesRegex(ValueError,'identity missing'):latency_worker.main()
 def test_launchers_reject_unreviewed_stage_before_gpu_or_container(self):
  import latency_launch,preflight_worker
  with patch('sys.argv',['latency_launch.py','--stage','smoke','--run-id','x']),patch.object(latency_launch,'admit',side_effect=ValueError('review missing')):
   with self.assertRaisesRegex(ValueError,'review missing'):latency_launch.main()
  with patch('sys.argv',['preflight_worker.py','--run-id','x']),patch.object(preflight_worker,'admit',side_effect=ValueError('review missing')):
   with self.assertRaisesRegex(ValueError,'review missing'):preflight_worker.main()

class ExecuteTests(unittest.TestCase):
 def make(self,fail=False):
  class FakeIds:
   shape=(1,1)
   def __init__(self,ids):self.ids=ids
   def detach(self):return self
   def cpu(self):return self
   def tolist(self):return [self.ids]
  class Model:
   generation_config=types.SimpleNamespace(eos_token_id=[2])
   def generate(self,*_args,**kw):
    kw['stopping_criteria'][0](FakeIds([5]),None)
    return FakeIds(([5]*31+[2]) if kw['max_new_tokens']==32 else [2])
  class Author:
   @staticmethod
   def generate_inner_transformers_fixed(w,_msg,dataset=None):
    w.model.generate(do_sample=False,max_new_tokens=2048,top_p=.001,top_k=1,temperature=.01,repetition_penalty=1.0)
    if fail:raise RuntimeError('decode failure')
    return 'answer'
  module=types.ModuleType('vlmeval.vlm.qwen3_vl.model_fixed_res');module.Qwen3VLChatCDPruner=Author
  packages={x:types.ModuleType(x) for x in ('vlmeval','vlmeval.vlm','vlmeval.vlm.qwen3_vl')};packages[module.__name__]=module
  class Adapter:
   def __init__(self,*_a):self.state={'prefill_seen':True}
   def __enter__(self):return self
   def __exit__(self,*_a):pass
  return types.SimpleNamespace(model=Model()),packages,Adapter
 def test_real_boundary_wiring_and_fixed_length(self):
  from runner import execute_one
  w,packages,adapter=self.make();sync=[];row={'image_start':1,'message':[],'task':'T'}
  with patch.dict(sys.modules,packages),patch('light_adapter.LightAdapter',adapter):result=execute_one(w,None,None,row,'C-ON','fixed32',lambda:sync.append(1))
  self.assertEqual(len(sync),3);self.assertEqual(result['output_tokens'],32);self.assertEqual(result['boundaries']['first_criteria_length'],1);self.assertEqual(result['generation_scalars']['max_new_tokens'],32)
 def test_generation_returned_then_decode_failed_is_preserved(self):
  from runner import execute_one
  w,packages,adapter=self.make(fail=True);partial=[];row={'image_start':1,'message':[],'task':'T'}
  with patch.dict(sys.modules,packages),patch('light_adapter.LightAdapter',adapter):
   with self.assertRaisesRegex(RuntimeError,'decode failure'):execute_one(w,None,None,row,'C-ON','natural',lambda:None,partial.append)
  self.assertEqual(partial[0]['output_ids'],[2]);self.assertEqual(partial[0]['error_type'],'RuntimeError')

class VerificationTests(unittest.TestCase):
 def case(self):
  job={'task':'T','qid':'1','group_id':'g','phase':'measure','protocol':'natural','mode':'C-ON','order':0}
  result={'mode':'C-ON','protocol':'natural','answer':'answer','output_ids':[2],'output_tokens':1,'eos_ids':[2],'eos':True,'truncated':False,
          'generation_scalars':{'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0},'min_new_tokens':None,
          'instrumented_logits_ready':False,'boundaries':{'t0_before_image_read_ns':100,'first_token_selected_ns':200,'generation_complete_ns':300,
          'first_criteria_length':1,'criteria_calls':1,'explicit_sync_boundaries':['before_t0','first_token','generation_end'],'logits_ready_ns':None,
          'time_namespace':'ns','clock_implementation':'clock_gettime(CLOCK_MONOTONIC)'}}
  events=[{'status':'generated','job':job,'worker_pid':123,'measurement_role':'latency_interleaved','result':result},
          {'status':'completed','job':job,'worker_pid':123,'measurement_role':'latency_interleaved'}]
  return {'schedule':[job]},events
 def test_validation_and_health_overlap_keeps_measurement(self):
  from verify_core import annotate_validated
  s,e=self.case();snap=[{'time_namespace':'ns','health_snapshot_window_monotonic_ns':[0,50]},
    {'time_namespace':'ns','health_snapshot_window_monotonic_ns':[150,160]},
    {'time_namespace':'ns','health_snapshot_window_monotonic_ns':[400,450],'event':'terminal_snapshot'}]
  started={'worker_pid':123,'snapshot_seconds':3600,'poll_seconds':15};terminal={'worker_pid':123,'reason':'completed','worker_returncode':0,'remaining_process_group':[]}
  out=annotate_validated(s,e,123,lambda ids:'answer',snap,'ns',started,terminal)
  self.assertEqual(out['n_generated'],1);self.assertEqual(out['n_monitor_overlaps'],1);self.assertTrue(out['events'][0]['monitor_overlap']);self.assertEqual(len(out['events']),2)
  with self.assertRaisesRegex(ValueError,'snapshot journal'):annotate_validated(s,e,123,lambda ids:'answer',snap[:-1],'ns',started,terminal)
 def test_bad_output_and_boundary_rejected(self):
  from verify_core import validate_events
  s,e=self.case();e[0]['result']['output_ids']=[1]*2048+[2]
  with self.assertRaisesRegex(ValueError,'generation recipe/output'):validate_events(s,e,123,lambda ids:'answer')
  s,e=self.case();e[0]['result']['boundaries']['first_criteria_length']=0
  with self.assertRaisesRegex(ValueError,'first-token timing boundary'):validate_events(s,e,123,lambda ids:'answer')
  s,e=self.case()
  with self.assertRaisesRegex(ValueError,'decoding mismatch'):validate_events(s,e,123,lambda ids:'wrong')
if __name__=='__main__':unittest.main()
