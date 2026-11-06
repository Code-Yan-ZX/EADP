import copy
import types
import unittest
import runner

class Adapter:
 def __init__(self,_torch,_wrapper,_native,_start,budget):self.state={'prefill_seen':True,'greedy_calls':0 if budget==1024 else 1}
 def __enter__(self):return self
 def __exit__(self,*args):pass

class Tests(unittest.TestCase):
 def setUp(self):
  self.events=[];self.raw=[];self.calls=[];self.clock_value=0;self.fail_final_sync=False;self.output=[10,9]
  self.recipe={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0}
  self.row={'task':'D','qid':'1','group_id':'g','message_sha256':'m','image_sha256':'i','image_start':4,'split':'DEV'}
  self.accepted=dict(self.row,budget=256,ds_on=True,generation_scalars=self.recipe,eos_ids=[9],output_ids=[10,9],answer='answer',eos=True,truncated=False)
  test=self
  class Output:
   def detach(self):test.events.append('detach');return self
   def cpu(self):test.events.append('cpu');return self
   def tolist(self):test.events.append('tolist');return [test.output]
  class Model:
   generation_config=types.SimpleNamespace(eos_token_id=9)
   def generate(self,**kw):
    test.calls.append(kw);test.events.append('generate')
    for n in range(1,len(test.output)+1):kw['stopping_criteria'][0](types.SimpleNamespace(shape=[1,n]),None)
    test.events.append('return');return Output()
  self.model=Model();self.wrapper=types.SimpleNamespace(model=self.model,pruner=types.SimpleNamespace())
  self.torch=types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=self.sync))
 def sync(self):
  self.events.append('sync')
  if self.fail_final_sync and self.events.count('sync')==3:raise RuntimeError('sync failed')
 def clock(self):self.clock_value+=10;self.events.append('clock');return self.clock_value
 def store(self,value):self.events.append('raw_fsync');self.raw.append(value)
 def validate(self,m):
  if 'generate' in vars(m):raise ValueError('instance override')
 def author(self,w,_row):self.events.append('image_read');w.model.generate(**self.recipe);self.events.append('decode');return 'answer'
 def execute(self,protocol='natural',author=None,accepted=None,budget=256,adapter=Adapter):
  return runner.execute(self.torch,self.wrapper,None,self.row,budget,protocol,accepted or self.accepted,adapter,list,author or self.author,self.store,self.validate,self.clock)
 def test_completion_before_materialization_and_journal(self):
  x=self.execute();self.assertEqual(x['output_ids'],[10,9]);self.assertEqual(x['boundaries']['criteria_calls'],2)
  self.assertEqual(self.events,['sync','clock','image_read','generate','sync','clock','return','sync','clock','detach','cpu','tolist','raw_fsync','decode'])
  self.assertNotIn('generate',vars(self.model));self.assertEqual(self.raw[0]['generation_complete_ns'],30)
 def test_fixed32_real_controls_and_boundary(self):
  self.output=list(range(100,132));x=self.execute('fixed32')
  self.assertEqual(self.calls[0]['min_new_tokens'],32);self.assertEqual(self.calls[0]['max_new_tokens'],32)
  self.assertNotIn('eos_token_id',self.calls[0]);self.assertNotIn('logits_processor',self.calls[0])
  self.assertEqual(x['boundaries']['criteria_calls'],32);self.assertEqual(x['eos_ids'],[9]);self.assertEqual(len(self.raw[0]['output_ids']),32)
 def test_fixed32_short_or_early_eos_is_preserved_then_rejected(self):
  for output in ([1,2],list(range(100,110))+[9]+list(range(111,132))):
   self.output=output
   with self.assertRaisesRegex(ValueError,'fixed32'):self.execute('fixed32')
   self.assertEqual(self.raw[-1]['output_ids'],output)
 def test_recipe_and_controls_rejected_before_actual_generation(self):
  for key in self.recipe:
   bad=dict(self.recipe);bad[key]='bad'
   with self.assertRaisesRegex(ValueError,'recipe mismatch'):self.execute(author=lambda w,_r:w.model.generate(**bad))
  with self.assertRaisesRegex(ValueError,'EOS mismatch'):self.execute(author=lambda w,_r:w.model.generate(**self.recipe,eos_token_id=[8]))
  with self.assertRaisesRegex(ValueError,'unexpected generation controls'):self.execute(author=lambda w,_r:w.model.generate(**self.recipe,logits_processor=[]))
  bad=dict(self.recipe,max_new_tokens=17)
  with self.assertRaisesRegex(ValueError,'original fixed32 recipe'):self.execute('fixed32',author=lambda w,_r:w.model.generate(**bad))
  self.assertEqual(self.calls,[])
 def test_natural_mismatch_and_answer_exception_preserve_raw(self):
  x=copy.deepcopy(self.accepted);x['output_ids']=[123,9]
  with self.assertRaisesRegex(ValueError,'output mismatch'):self.execute(accepted=x)
  self.assertEqual(self.raw[-1]['output_ids'],[10,9])
  def bad_decode(w,row):self.author(w,row);raise RuntimeError('decoder failed')
  with self.assertRaisesRegex(RuntimeError,'decoder failed'):self.execute(author=bad_decode)
  self.assertEqual(self.raw[-1]['output_ids'],[10,9]);self.assertNotIn('generate',vars(self.model))
 def test_sync_failure_saves_failed_raw_without_fake_end(self):
  self.fail_final_sync=True
  with self.assertRaisesRegex(RuntimeError,'sync failed'):self.execute()
  self.assertIsNone(self.raw[-1]['generation_complete_ns']);self.assertIn('sync failed',self.raw[-1]['boundary_error'])
  self.assertEqual(self.raw[-1]['output_ids'],[10,9]);self.assertNotIn('generate',vars(self.model))
 def test_repeated_and_keep_all_paths(self):
  for _ in range(2):self.execute();self.assertNotIn('generate',vars(self.model))
  self.execute(budget=1024,accepted=dict(self.accepted,budget=1024))
  class Bad(Adapter):
   def __init__(self,*args):super().__init__(*args);self.state['greedy_calls']=1
  with self.assertRaisesRegex(ValueError,'path incomplete'):self.execute(budget=1024,accepted=dict(self.accepted,budget=1024),adapter=Bad)
  self.assertEqual(self.raw[-1]['output_ids'],[10,9])
 def test_instance_override_is_rejected_before_generation(self):
  self.model.generate=lambda **kw:None
  with self.assertRaisesRegex(ValueError,'instance override'):self.execute()
  self.assertEqual(self.calls,[]);self.assertEqual(self.raw,[])

if __name__=='__main__':unittest.main()
