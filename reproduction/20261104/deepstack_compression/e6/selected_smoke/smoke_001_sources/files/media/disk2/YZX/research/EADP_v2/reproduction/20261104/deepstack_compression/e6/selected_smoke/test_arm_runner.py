import json,sys,tempfile,types,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import arm_runner

class Model:
 def __init__(self):
  self.model=SimpleNamespace(rope_deltas=None);self.generation_config=SimpleNamespace(eos_token_id=[2]);self.calls=0
 def generate(self,**kw):self.calls+=1;return [[2]]
class Wrapper:
 def __init__(self):
  self.model=Model();self.pruner=Pruner();self.processor=SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a,**k:[7]));self.wrong=False
 def generate_inner_transformers_fixed(self,message,dataset=None):
  cfg={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.}
  if self.wrong:cfg['do_sample']=True
  self.model.generate(**cfg);return ''
class Pruner:pass
class Torch:
 @staticmethod
 def inference_mode():
  from contextlib import nullcontext
  return nullcontext()
class KV:
 def __init__(self,*_):self.fingerprints=['fake']
 def __enter__(self):return self
 def __exit__(self,*_):pass
class Contract:
 def validate_cache(self,*a):pass
 def compare_pair(self,*a):pass
 def compare_stock_forced(self,*a):pass
 def compare_forced(self,*a):pass
class Stock:
 def run_stock(self,torch,w,method,row,generate,callback,archive,scalars,token):
  result={'method':method,'path':'author_stock','task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],
          'forced':token is not None,'output_ids':[2]};callback(result);return result
def baseline(torch,w,method,row,generate,callback,archive,forced_token=None,capture_prefill_logits=False):
 original=w.model.generate
 if forced_token is not None:
  def force(**kw):
   kw.update(max_new_tokens=33,min_new_tokens=33,eos_token_id=[],logits_processor=['fixed']);return original(**kw)
  w.model.generate=force
 try:generate(w,row)
 finally:w.model.generate=original
 result={'method':method,'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],
         'forced':forced_token is not None,'output_ids':[2]};callback(result);return result

class ArmTests(unittest.TestCase):
 def setup_case(self):
  rows=[{'task':task,'qid':str(i),'group_id':'g'+str(i),'message':[]} for i,task in enumerate(('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'))]
  recipe={'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.}
  return Wrapper(),rows,recipe
 def test_runs_all_four_pairs_and_durable_output_markers(self):
  w,rows,recipe=self.setup_case()
  with tempfile.TemporaryDirectory() as tmp,patch.object(arm_runner,'prepare',return_value=(Stock(),Contract())),patch.object(arm_runner,'CacheFingerprints',KV),patch.dict(sys.modules,{'baseline_runner':SimpleNamespace(run_arm=baseline)}):
   result=arm_runner.run(Torch,w,Wrapper,Pruner,'div',rows,recipe,tmp)
   self.assertEqual(result['generations'],8);self.assertEqual(len(Path(tmp,'generated.jsonl').read_text().splitlines()),8)
   self.assertEqual(len(Path(tmp,'records.jsonl').read_text().splitlines()),8);self.assertEqual(w.model.calls,4)
 def test_wrong_actual_recipe_never_reaches_generate(self):
  w,rows,recipe=self.setup_case();w.wrong=True
  with tempfile.TemporaryDirectory() as tmp,patch.object(arm_runner,'prepare',return_value=(Stock(),Contract())),patch.object(arm_runner,'CacheFingerprints',KV),patch.dict(sys.modules,{'baseline_runner':SimpleNamespace(run_arm=baseline)}):
   with self.assertRaisesRegex(ValueError,'before generate'):arm_runner.run(Torch,w,Wrapper,Pruner,'div',rows,recipe,tmp)
   self.assertEqual(w.model.calls,0);self.assertEqual(len(Path(tmp,'generated.jsonl').read_text().splitlines()),1)
   events=[json.loads(s) for s in Path(tmp,'events.jsonl').read_text().splitlines()]
   self.assertEqual(events[-1]['event'],'arm_started');self.assertEqual(events[-1]['path'],'adapted')
 def test_rejects_substituted_author_pruner(self):
  w,rows,recipe=self.setup_case();w.pruner=object()
  with tempfile.TemporaryDirectory() as tmp,patch.object(arm_runner,'prepare',return_value=(Stock(),Contract())),patch.dict(sys.modules,{'baseline_runner':SimpleNamespace(run_arm=baseline)}):
   with self.assertRaisesRegex(ValueError,'exact imported'):arm_runner.run(Torch,w,Wrapper,Pruner,'div',rows,recipe,tmp)
if __name__=='__main__':unittest.main()
