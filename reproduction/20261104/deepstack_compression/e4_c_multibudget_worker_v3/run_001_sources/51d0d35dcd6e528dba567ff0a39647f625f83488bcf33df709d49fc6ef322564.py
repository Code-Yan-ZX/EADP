import copy,importlib.util,types,unittest
from pathlib import Path
import runner
BASE=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('_budget_timing_timer',BASE/'e4_c_latency/timer.py');timer=importlib.util.module_from_spec(spec);spec.loader.exec_module(timer)
class IDs:
 def detach(self):return self
 def cpu(self):return self
 def tolist(self):return [[10,9]]
class Adapter:
 def __init__(self,_torch,_wrapper,_native,_start,budget):self.state={'prefill_seen':True,'greedy_calls':0 if budget==1024 else 1}
 def __enter__(self):return self
 def __exit__(self,*_args):pass
class Tests(unittest.TestCase):
 def setUp(self):
  self.calls=[];self.raw=[];self.sync=[]
  self.recipe={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0}
  self.row={'task':'D','qid':'1','group_id':'g','message_sha256':'m','image_sha256':'i','image_start':4,'split':'DEV'}
  self.accepted=dict(self.row,budget=256,ds_on=True,generation_scalars=self.recipe,eos_ids=[9],output_ids=[10,9],answer='answer',eos=True,truncated=False)
  def model_generate(**kwargs):
   self.calls.append(kwargs)
   for n in (1,2):kwargs['stopping_criteria'][0](types.SimpleNamespace(shape=[1,n]),None)
   return IDs()
  self.model=types.SimpleNamespace(generate=model_generate,generation_config=types.SimpleNamespace(eos_token_id=9))
  self.wrapper=types.SimpleNamespace(model=self.model,pruner=types.SimpleNamespace())
  self.torch=types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda:self.sync.append(1)))
 def author(self,w,_row):w.model.generate(**self.recipe);return 'answer'
 def execute(self,author=None,accepted=None,budget=256,adapter=Adapter):
  return runner.execute(self.torch,self.wrapper,None,self.row,budget,accepted or self.accepted,adapter,timer.measure,list,author or self.author,self.raw.append,lambda _model:None)
 def test_actual_first_token_boundary_and_raw(self):
  x=self.execute();self.assertEqual(x['output_ids'],[10,9]);self.assertEqual(x['boundaries']['first_criteria_length'],1)
  self.assertEqual(x['boundaries']['criteria_calls'],2);self.assertEqual(len(self.sync),3);self.assertEqual(self.raw[0]['output_ids'],[10,9]);self.assertIs(self.model.generate.__func__ if hasattr(self.model.generate,'__func__') else self.model.generate,self.model.generate)
 def test_wrong_recipe_or_eos_cannot_call_model(self):
  for key in self.recipe:
   bad=dict(self.recipe);bad[key]='bad'
   with self.assertRaisesRegex(ValueError,'recipe mismatch'):
    self.execute(author=lambda w,_r:w.model.generate(**bad))
   self.assertEqual(self.calls,[])
  with self.assertRaisesRegex(ValueError,'EOS mismatch'):
   self.execute(author=lambda w,_r:w.model.generate(**self.recipe,eos_token_id=[10]))
  self.assertEqual(self.calls,[])
 def test_output_mismatch_preserves_raw(self):
  accepted=copy.deepcopy(self.accepted);accepted['output_ids']=[42,9]
  with self.assertRaisesRegex(ValueError,'output mismatch'):self.execute(accepted=accepted)
  self.assertEqual(self.raw[0]['output_ids'],[10,9])
 def test_keepall_cannot_call_greedy(self):
  class Bad(Adapter):
   def __init__(self,*args):super().__init__(*args);self.state['greedy_calls']=1
  accepted=dict(self.accepted,budget=1024)
  with self.assertRaisesRegex(ValueError,'method path'):self.execute(accepted=accepted,budget=1024,adapter=Bad)
  self.assertEqual(self.raw[0]['output_ids'],[10,9])
 def test_patched_instance_rejected_before_real_runner_calls(self):
  import generation_identity
  with self.assertRaisesRegex(ValueError,'implementation changed'):
   runner.execute(self.torch,self.wrapper,None,self.row,256,self.accepted,Adapter,timer.measure,list,self.author,self.raw.append,generation_identity.validate_model_class)
  self.assertEqual(self.calls,[]);self.assertEqual(self.raw,[])
 def test_repeated_jobs_restore_original_class_descriptor(self):
  calls=self.calls
  class Model:
   generation_config=types.SimpleNamespace(eos_token_id=9)
   def generate(self,**kwargs):
    calls.append(kwargs)
    for n in (1,2):kwargs['stopping_criteria'][0](types.SimpleNamespace(shape=[1,n]),None)
    return IDs()
  model=Model();self.wrapper.model=model
  def validate(m):
   if 'generate' in vars(m):raise ValueError('instance override')
  for _ in range(2):
   runner.execute(self.torch,self.wrapper,None,self.row,256,self.accepted,Adapter,timer.measure,list,self.author,self.raw.append,validate)
   self.assertNotIn('generate',vars(model))
  self.assertEqual(len(calls),2)
