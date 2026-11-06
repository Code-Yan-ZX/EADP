"""Real worker.main with synthetic dependencies only: never CUDA or predictions."""
import contextlib,hashlib,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import model_worker,model_core,diagnostic_run,inventory
class FakeCuda:
 def is_initialized(self):return False
 def get_device_name(self,n):return 'NVIDIA A40'
 def set_per_process_memory_fraction(self,*a):pass
 def max_memory_allocated(self):return 0
 def max_memory_reserved(self):return 0
class Tests(unittest.TestCase):
 def execute(self,preflight=False,fail=False,reject=None):
  rows=[{'task':task,'qid':str(i),'group_id':str(i),'image_sha256':str(i)*64,'message_sha256':'m','split':'DEV'} for i,task in enumerate(('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'))]
  choices={(r['task'],r['qid'],seed):{**r,'seed':seed,'spatial_ids':list(range(256)),'derived_seed64':seed} for r in rows for seed in inventory.SEEDS}
  references={(r['task'],r['qid']):[{'arm':'P3-D1','forced':False}] for r in rows}
  prep={'rows':rows,'refs':{(r['task'],r['qid']):{} for r in rows},'plan':{'recipe':{}},'weights_before':{}}
  torch=types.SimpleNamespace(cuda=FakeCuda(),bfloat16='bf16',inference_mode=contextlib.nullcontext)
  lm=types.SimpleNamespace(register_forward_hook=lambda *a,**k:types.SimpleNamespace(remove=lambda:None))
  wrapper=types.SimpleNamespace(visual_token_num=256,pruner=types.SimpleNamespace(visual_token_num=256),model=types.SimpleNamespace(dtype='bf16',config=types.SimpleNamespace(_attn_implementation='flash_attention_2'),model=types.SimpleNamespace(language_model=lm)))
  class Adapter:pass
  class Qwen:generate_inner_transformers_fixed=lambda *a,**kw:''
  def fake_arm(_torch,_w,_native,row,seed,ref,sel,runner,adapter,generate,raw,archive,identity,cache,forced):
   result={k:row[k] for k in ('task','qid','group_id')};result.update(seed=seed,forced=forced,spatial_ids=list(range(256)),derived_seed64=seed,output_ids=[100],actual_injection_verified=True);raw(result)
   if fail:raise RuntimeError('after exposure')
   return result
  def admit(_):
   if reject=='gate':raise ValueError('gate refused')
   return {'approved':True}
  gate=types.SimpleNamespace(admit=admit,plan=lambda:{'source_sha256':{}},sha=lambda p:('f'*64 if Path(p).name.startswith('execution_plan') else hashlib.sha256(Path(p).read_bytes()).hexdigest()),read=lambda p: {'anchors':rows} if Path(p).name=='plan001.json' else json.loads(Path(p).read_text()))
  modules={'_random_model_gate':gate,'_random_latency_worker':types.SimpleNamespace(prepare=lambda x:prep,load_runtime=lambda x:(torch,None,lambda **kwargs:wrapper,{'test':'fake'}),preflight_one=lambda *a:None),'_random_e1_admission':types.SimpleNamespace(validate=lambda **kw:object()),'_random_e1_refs':types.SimpleNamespace(load=lambda *a:references),'_random_manifest':types.SimpleNamespace(load=lambda *a:choices),'_random_inventory':inventory,'_random_identity_contract':types.SimpleNamespace(validate_e1_identity=lambda *a:True),'_random_smoke_core':types.SimpleNamespace(run_arm=fake_arm),'_random_diagnostic_run':diagnostic_run,'_random_model_core':model_core,'_random_selector':types.SimpleNamespace(selection=lambda image,seed:{'spatial_ids':list(range(256)),'derived_seed64':seed}),'_random_model_adapter':types.SimpleNamespace(RandomAdapter=Adapter),'_random_budget_runner':object(),'_random_fixed_common':types.SimpleNamespace(CacheFingerprints=object)}
  def load(name,path):return modules[name]
  qwen=types.ModuleType('vlmeval.vlm.qwen3_vl.model_fixed_res');qwen.Qwen3VLChatCDPruner=Qwen
  selector=types.ModuleType('selector');selector.__file__=str(model_worker.RB/'selector.py');cid='a'*64
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);(out/'launch.json').write_text(json.dumps({'approval_bundle':{'approved':True},'worker_sha256':gate.sha(model_worker.__file__),'run_out':str(out)}));(out/'container_identity.json').write_text(json.dumps({'container_id':cid,'host_cgroup':('wrong' if reject=='container' else cid),'host_init':{'start_ticks':1}}))
   original=Path.read_text
   def read_text(path,*a,**kw):return cid if str(path)=='/proc/self/cgroup' else original(path,*a,**kw)
   argv=['model_worker.py','--out',str(out)]+(['--preflight-only'] if preflight else [])
   with patch.object(model_worker,'load',side_effect=load),patch.object(model_worker.inspect,'getfile',return_value=str(model_worker.RB/'adapter.py')),patch.object(Path,'read_text',read_text),patch.object(sys,'argv',argv),patch.dict(sys.modules,{'selector':selector,'vlmeval.vlm.qwen3_vl.model_fixed_res':qwen}):
    if reject:
     with self.assertRaises(ValueError):model_worker.main()
     self.assertFalse((out/'generated.jsonl').exists());self.assertFalse((out/'worker_complete.json').exists())
    elif fail:
     with self.assertRaisesRegex(RuntimeError,'after exposure'):model_worker.main()
     self.assertFalse((out/'worker_complete.json').exists());self.assertEqual(json.loads((out/'generated.jsonl').read_text())['raw']['output_ids'],[100]);self.assertTrue((out/'failure.json').exists())
    else:
     model_worker.main()
     if preflight:self.assertFalse(json.loads((out/'worker_preflight.json').read_text())['cuda_initialized'])
     else:self.assertEqual(json.loads((out/'worker_complete.json').read_text())['arms'],18);self.assertEqual(len((out/'generated.jsonl').read_text().splitlines()),18)
 def test_missing_admission_refuses_before_generation(self):self.execute(reject='gate')
 def test_container_mismatch_refuses_before_generation(self):self.execute(reject='container')
 def test_preflight_path(self):self.execute(preflight=True)
 def test_full_fake_18_arms(self):self.execute()
 def test_failure_after_generation_is_preserved(self):self.execute(fail=True)
if __name__=='__main__':unittest.main()
