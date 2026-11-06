import copy,json,sys,tempfile,time,types,unittest
from pathlib import Path
from unittest.mock import patch
import batch_gate,runtime_worker,preflight
class GateTests(unittest.TestCase):
 def test_budget_rejects_exhausted_or_altered_contract(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);initial={'budget_start_unix':1793806214,'budget_end_unix':1794065414};(root/'initial_plan.json').write_text(json.dumps(initial))
   plan={**initial,'source_sha256':{str(root/'initial_plan.json'):batch_gate.sha(root/'initial_plan.json')},'inner_timeout_seconds':7200,'outer_timeout_seconds':7380,'snapshot_seconds':3600,'progress_stale_seconds':2400,'question_checkpoint_seconds':1800,'questions':128,'new_generations_per_question':6,'maximum_new_generations':768}
   with patch.object(batch_gate,'ROOT',root):
    batch_gate.budget(plan,1793806215)
    with self.assertRaises(ValueError):batch_gate.budget(plan,1794065414-7379)
    bad=copy.deepcopy(plan);bad['questions']=127
    with self.assertRaises(ValueError):batch_gate.budget(bad,1793806215)
 def test_cached_wrong_module_rejected_before_import(self):
  fake=types.ModuleType('e1_admission');fake.__file__='/tmp/unrelated.py'
  with patch.dict(sys.modules,{'e1_admission':fake}),self.assertRaisesRegex(ValueError,'conflicting cached module'):
   runtime_worker.reject_wrong_cached_modules()
 def test_absent_approval_locked(self):
  with tempfile.TemporaryDirectory() as d,patch.object(batch_gate,'APPROVAL',Path(d)/'missing'):
   for pre in (False,True):
    with self.assertRaisesRegex(ValueError,'batch-2 locked'):batch_gate.validate(preflight_only=pre)
 def test_actual_mounts_and_environment(self):
  cmd=['docker','run','--mount','type=bind,src=/tmp,dst=/data,readonly','--env','CUDA_VISIBLE_DEVICES=','image']
  info={'Mounts':[{'Type':'bind','Source':'/tmp','Destination':'/data','RW':False}],'Config':{'Env':['PATH=/bin','CUDA_VISIBLE_DEVICES=']}}
  self.assertTrue(preflight.actual_mounts_and_env_match(cmd,info))
  for mutation in ('rw','extra','env','source'):
   b=copy.deepcopy(info)
   if mutation=='rw':b['Mounts'][0]['RW']=True
   if mutation=='extra':b['Mounts'].append({'Type':'bind','Source':'/','Destination':'/extra','RW':True})
   if mutation=='env':b['Config']['Env'][-1]='CUDA_VISIBLE_DEVICES=0'
   if mutation=='source':b['Mounts'][0]['Source']='/'
   self.assertFalse(preflight.actual_mounts_and_env_match(cmd,b),mutation)
 def test_positive_review_gate_and_tampered_review(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);here=root/'batch';here.mkdir();revdir=root/'e1_independent_review';revdir.mkdir()
   cpu_script=here/'cpu_validation.py';cpu_script.write_text('cpu')
   reviewer=revdir/'review_m3batch2_preflight.py';reviewer.write_text('reviewer')
   sample=here/'sample.json';sample.write_text('{}')
   plan=here/'execution_plan_002.json';plan.write_text(json.dumps({'source_sha256':{},'sample_path':str(sample),'sample_sha256':batch_gate.sha(sample)}))
   src={str(p):batch_gate.sha(p) for p in (cpu_script,plan)}
   cpu=here/'cpu.json';cpu.write_text(json.dumps({'status':'passed','GT_opened':False,'predictions_opened':False,'real_model_loaded':False,'source_sha256_before':src,'source_sha256_after':src}))
   def supervisor(name,script,out):
    start=here/(name+'_start.json');end=here/(name+'_end.json')
    start.write_text(json.dumps({'command':['python',str(script),'--out',str(out)],'cwd':str(root),'worker_pid':123}))
    end.write_text(json.dumps({'reason':'completed','worker_returncode':0,'remaining_process_group':[],'worker_pid':123}))
    return start,end
   cs,ct=supervisor('cpu',cpu_script,cpu.parent)
   rp=here/'independent_preflight_review.json';rs,rt=supervisor('review',reviewer,rp)
   review={'status':'passed','independent_review':True,'preflight_only':True,'reviewer_sha256':batch_gate.sha(reviewer),'cpu_proof_sha256':batch_gate.sha(cpu),'cpu_started_sha256':batch_gate.sha(cs),'cpu_terminal_sha256':batch_gate.sha(ct),'plan_sha256':batch_gate.sha(plan),'source_sha256':src,'GT_opened':False,'gpu_used':False}
   rp.write_text(json.dumps(review));approval=here/'runtime_approval.json';approval.write_text(json.dumps({'cpu_proof':str(cpu),'cpu_started':str(cs),'cpu_terminal':str(ct),'review_proof':str(rp),'review_started':str(rs),'review_terminal':str(rt)}))
   with patch.object(batch_gate,'HERE',here),patch.object(batch_gate,'ROOT',root),patch.object(batch_gate,'APPROVAL',approval),patch.object(batch_gate,'budget'):
    self.assertEqual(batch_gate.validate(preflight_only=True)['source_count_verified'],2)
    for field in ('independent_review','preflight_only'):
     bad={**review,field:False};rp.write_text(json.dumps(bad))
     with self.assertRaises(ValueError):batch_gate.validate(preflight_only=True)
    rp.write_text(json.dumps(review));cpu_script.write_text('tampered')
    with self.assertRaises(ValueError):batch_gate.validate(preflight_only=True)
if __name__=='__main__':unittest.main()
