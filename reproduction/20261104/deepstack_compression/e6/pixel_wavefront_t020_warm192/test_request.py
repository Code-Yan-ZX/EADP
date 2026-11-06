"""CPU fake model executes request path and rejects cache reuse."""
import contextlib,importlib.util,types,unittest
from pathlib import Path
import torch
import request
B=Path(__file__).resolve().parents[2];p=B/'e6/pixel_wavefront_t020_model_timing_v2/t020_timing_boundaries.py';sp=importlib.util.spec_from_file_location('_warm192_timer_test',p);timer=importlib.util.module_from_spec(sp);sp.loader.exec_module(timer)
class Selector:
 method='max';threshold=.2;anchored=True
 def _select_2d_loco_anchored(self,*args):return torch.tensor([2,5])
class FakeCache:
 def __init__(self,bad=False):self.bad=bad
 def get_seq_length(self,i):return 2 if self.bad and i==13 else 0
class Inner(torch.nn.Module):
 def __init__(self):super().__init__();self.rope_deltas='previous';self.seen_reset=[]
 def forward(self,**kw):self.seen_reset.append(self.rope_deltas is None);self.rope_deltas='new';return None
class FakeModel:
 def __init__(self,patch):self.model=Inner();self.generation_config=types.SimpleNamespace(eos_token_id=[9]);self.config=types.SimpleNamespace(image_token_id=42);self.patch=patch;self.bad=False
 def generate(self,**kw):
  self.patch.compute_merged_keep_indices();self.model(past_key_values=FakeCache(self.bad),cache_position=torch.tensor([0,1]));out=torch.tensor([[1,2,9]]);kw['stopping_criteria'](out,None);return out
class Wrapper:
 def __init__(self,m):self.model=m;self.generate_kwargs={k:v for k,v in request.RECIPE.items() if k not in ('do_sample','num_beams','use_cache')}
 def _processor_inputs(self,*args):return {'input_ids':torch.tensor([[1,2]])}
 def generate_inner_transformers_fixed(self,message,dataset):
  inp=self._processor_inputs(message);self.model.generate(**inp,do_sample=False,**self.generate_kwargs);return 'fake'
@contextlib.contextmanager
def replaced(obj,name,fn):
 had=name in obj.__dict__;stored=obj.__dict__.get(name);old=getattr(obj,name);setattr(obj,name,fn(old))
 try:yield
 finally:
  if had:setattr(obj,name,stored)
  else:delattr(obj,name)
@contextlib.contextmanager
def count_calls(module,name,kind,expected):
 old=getattr(module,name);counts=[]
 def call(*a,**kw):
  v=old(*a,**kw);counts.append(len(v[0]));return v
 setattr(module,name,call)
 try:
  yield counts
  if len(counts)!=expected:raise ValueError('count')
 finally:setattr(module,name,old)
class RequestTest(unittest.TestCase):
 def setUp(self):
  self.patch=types.SimpleNamespace(compute_merged_keep_indices=lambda:[Selector()._select_2d_loco_anchored(None,1,1,None)]);self.model=FakeModel(self.patch);self.wrapper=Wrapper(self.model);self.base=types.SimpleNamespace(replaced=replaced,count_calls=count_calls);self.fake_torch=types.SimpleNamespace(inference_mode=torch.inference_mode,cuda=types.SimpleNamespace(synchronize=lambda:None));self.row={'task':'fake','qid':'1','group_id':'g','message':[]};self.ref={'processor':{'x':1},'image_start':1};self.raw=[]
 def run_request(self,arm):
  return request.execute(self.wrapper,self.patch,self.fake_torch,self.row,self.ref,Selector,lambda *a:torch.tensor([2,6]),arm,self.base,timer,lambda *a:{'processor':self.ref['processor']},self.raw.append)
 def test_two_independent_requests_and_differences_preserved(self):
  original=Selector._select_2d_loco_anchored;a=self.run_request('author');b=self.run_request('wavefront')
  self.assertEqual(a['selected_indices'],[2,5]);self.assertEqual(b['selected_indices'],[2,6]);self.assertEqual(a['first_token_id'],9);self.assertTrue(a['first_token_is_eos']);self.assertEqual(len(self.raw),2);self.assertEqual(self.model.model.seen_reset,[True,True]);self.assertEqual(len(self.model.model._forward_pre_hooks),0);self.assertIs(Selector._select_2d_loco_anchored,original)
 def test_nonempty_one_layer_rejected_and_hooks_cleaned(self):
  original=Selector._select_2d_loco_anchored;self.model.bad=True
  with self.assertRaisesRegex(ValueError,'nonempty initial KV'):self.run_request('author')
  self.assertEqual(self.raw,[]);self.assertEqual(len(self.model.model._forward_pre_hooks),0);self.assertIs(Selector._select_2d_loco_anchored,original)
if __name__=='__main__':unittest.main()
