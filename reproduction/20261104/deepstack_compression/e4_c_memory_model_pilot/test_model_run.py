import unittest
from model_run import raw_capture,validate_result
class FakeTensor:
 def __init__(self,values):self.values=values
 def detach(self):return self
 def cpu(self):return self
 def tolist(self):return [self.values]
class Tests(unittest.TestCase):
 def test_exact_anchor_and_identity_mismatch(self):
  ref={'output_ids':[1],'generation_scalars':{},'eos_ids':[2],'answer':'x','ordered_ids':[3],'processor':{k:'ok' for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')}}
  class V:
   def detach(self):return self
   def cpu(self):return self
   def tolist(self):return [3]
  r={'seen':{'output_ids':[1],'generation_scalars':{},'eos_ids':[2]},'answer':'x','ordered':V(),'inputs':{k:1 for k in ref['processor']}}
  self.assertEqual(validate_result(r,ref,lambda x:'ok')['status'],'passed');r['seen']['output_ids']=[9]
  with self.assertRaises(ValueError):validate_result(r,ref,lambda x:'ok')
 def test_journals_raw_before_postprocessing_and_rejects_recipe(self):
  calls=[];events=[];recipe={'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0};args={**recipe,'do_sample':False,'max_new_tokens':2048};out=FakeTensor([5,6]);fn,seen=raw_capture(lambda **kw:(calls.append('generate') or out),events.append,{'task':'t'},recipe,[3],[3]);self.assertIs(fn(**args),out);self.assertEqual(events[0]['output_ids'],[5,6]);self.assertEqual(seen['output_ids'],[5,6]);self.assertEqual(calls,['generate'])
  fn,_=raw_capture(lambda **kw:self.fail('must not generate'),events.append,{},recipe,[3],[4])
  with self.assertRaises(ValueError):fn(**args)
  fn,_=raw_capture(lambda **kw:self.fail('must not generate'),events.append,{},recipe,[3],[3]);args['top_k']=2
  with self.assertRaises(ValueError):fn(**args)
if __name__=='__main__':unittest.main(verbosity=2)
