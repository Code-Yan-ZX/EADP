import types,unittest
from smoke_core import run_arm
class Cache:
 def __init__(self,w):self.fingerprints=[[{k:{'dtype':'torch.bfloat16','shape':[1,8,256+i,128],'sha256':'a'*64} for k in ('key','value')} for _ in range(36)] for i in range(33)]
 def __enter__(self):return self
 def __exit__(self,*a):pass
class Tests(unittest.TestCase):
 def fixture(self):
  expected={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.};s={'task':'DocVQA_VAL','qid':'1','group_id':'g','split':'DEV','image_sha256':'a'*64,'message_sha256':'b'*64};e={**s,'arm':'P3-D1','forced':False,'ordered_ids':list(range(256)),'generation_scalars':expected,'eos_ids':[999]};return s,e
 def test_factory_passes_original_greedy_and_restores(self):
  s,e=self.fixture();model=types.SimpleNamespace(generate=lambda **kw:[1],generation_config=types.SimpleNamespace(eos_token_id=[999]));w=types.SimpleNamespace(model=model);prior=object();runner=types.SimpleNamespace(BudgetAdapter=prior);calls=[];ids=list(range(256))
  def runner_arm(t,w,p,s,b,d,g,j,archive,forced_token=None,capture_prefill_logits=False):
   self.assertIsNot(runner.BudgetAdapter,prior);runner.BudgetAdapter(t,w,p,b,d);g(w,s);j({'output_ids':[1,999]});return {'selection_mode':'e6_random_fixed_seed','ordered_ids':ids,'spatial_ids':ids,'output_ids':[1,999]}
  runner.run_arm=runner_arm
  def adapter(*args):calls.append(args);return object()
  result=run_arm(None,w,None,s,20261104,e,lambda i,z:{'spatial_ids':ids,'derived_seed64':1},runner,adapter,lambda w,r:w.model.generate(**e['generation_scalars']),lambda x:calls.append(x),None,lambda *a:True,None)
  self.assertIs(runner.BudgetAdapter,prior);self.assertEqual(calls[0][-1],e['ordered_ids']);self.assertEqual(calls[1]['seed'],20261104);self.assertFalse(result['timing_eligible'])
 def test_forced33_and_mutated_controls(self):
  s,e=self.fixture();model=types.SimpleNamespace(generate=lambda **kw:[1],generation_config=types.SimpleNamespace(eos_token_id=[999]));w=types.SimpleNamespace(model=model);prior=object();runner=types.SimpleNamespace(BudgetAdapter=prior);ids=list(range(256));journal=[]
  def arm(t,w,p,s,b,d,g,j,archive,forced_token=None,capture_prefill_logits=False):
   self.assertEqual(forced_token,100);g(w,s);j({'output_ids':[100]*33});return {'selection_mode':'e6_random_fixed_seed','ordered_ids':ids,'spatial_ids':ids,'output_ids':[100]*33,'eos_ids':[],'decode_trace':[{}]*33,'retained_ids':list(range(256))}
  runner.run_arm=arm;selection=lambda i,z:{'spatial_ids':ids,'derived_seed64':1}
  result=run_arm(None,w,None,s,20261104,e,selection,runner,lambda *a:None,lambda w,r:w.model.generate(**e['generation_scalars']),journal.append,None,lambda *a:True,Cache,True)
  self.assertEqual(result['protocol'],'forced33');self.assertEqual(journal[0]['output_ids'],[100]*33)
  def wrong(w,r):return w.model.generate(**dict(e['generation_scalars'],top_k=10))
  with self.assertRaisesRegex(ValueError,'controls'):run_arm(None,w,None,s,20261104,e,selection,runner,lambda *a:None,wrong,journal.append,None,lambda *a:True,Cache,True)
  self.assertIs(runner.BudgetAdapter,prior)
 def test_e1_input_mismatch_rejected_before_runner(self):
  s,e=self.fixture();e['group_id']='bad'
  with self.assertRaisesRegex(ValueError,'E1 input'):run_arm(None,None,None,s,20261104,e,None,None,None,None,None,None,None,None)
if __name__=='__main__':unittest.main(verbosity=2)
