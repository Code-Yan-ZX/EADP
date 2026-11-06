import unittest
from contextlib import contextmanager
import qwen_request as q

class Tensor:
 shape=(1,4);dtype='int64';device='cpu'
class Config:
 eos_token_id=[151645,151643]
 def to_dict(self):return {**q.RECIPE,'eos_token_id':self.eos_token_id}
class Core:rope_deltas=None
class Model:
 def __init__(self):self.model=Core();self.generation_config=Config()
 def _prepare_generation_config(self,*a,**kw):return Config(),{}
 def generate(self,**kw):
  self._prepare_generation_config(None,**kw)
  kw['stopping_criteria'][0](Tensor(),None)
  return [1,2,3]
class Wrapper:
 def __init__(self):self.model=Model()
 def _processor_inputs(self):return {'input_ids':Tensor()}
class Module:QWEN3_FIXED_RESOLUTION=1024
class CUDA:
 @staticmethod
 def synchronize():pass
class Torch:cuda=CUDA()
@contextmanager
def fixed(module,value):
 module.QWEN3_FIXED_RESOLUTION=value
 try:yield
 finally:module.QWEN3_FIXED_RESOLUTION=1024
class Adapter:
 def __init__(self,*args):self.state={'prefill_seen':True,'greedy_calls':1}
 def __enter__(self):return self
 def __exit__(self,*args):return False
class TestActualRequestContract(unittest.TestCase):
 def test_both_arms_capture_real_boundary_and_restore(self):
  row={'task':'t','qid':'1','group_id':'g','image_sha256':'i','message_sha256':'m','image_start':4}
  accepted={**row,'generation_scalars':q.RECIPE,'eos_ids':[151645,151643]}
  for arm in ('eadp','rres'):
   w=Wrapper();saved=[]
   def author(wrapper,row):
    if arm=='rres':wrapper._processor_inputs()
    wrapper.model.generate(**q.RECIPE)
    return 'response'
   result=q.request(arm=arm,torch=Torch(),wrapper=w,module=Module,pruner=None,row=row,accepted=accepted,adapter_type=Adapter,criteria_type=list,fixed_resolution=fixed,author_generate=author,persist=saved.append)
   self.assertEqual(saved,[[1,2,3]])
   self.assertEqual(result['generation']['effective_generation_config']['top_k'],1)
   self.assertTrue(result['times']['raw_persisted_after_all_endpoints'])
   self.assertNotIn('generate',vars(w.model));self.assertNotIn('_processor_inputs',vars(w))
 def test_bad_kwargs_do_not_invoke_bottom_generate(self):
  row={'task':'t','qid':'1','group_id':'g','image_sha256':'i','message_sha256':'m','image_start':4}
  accepted={**row,'generation_scalars':q.RECIPE,'eos_ids':[151645,151643]};w=Wrapper();saved=[]
  def author(wrapper,row):
   wrapper._processor_inputs();kw={**q.RECIPE,'top_k':2};return wrapper.model.generate(**kw)
  with self.assertRaisesRegex(ValueError,'kwargs'):
   q.request(arm='rres',torch=Torch(),wrapper=w,module=Module,pruner=None,row=row,accepted=accepted,adapter_type=Adapter,criteria_type=list,fixed_resolution=fixed,author_generate=author,persist=saved.append)
  self.assertEqual(saved,[]);self.assertNotIn('generate',vars(w.model));self.assertEqual(Module.QWEN3_FIXED_RESOLUTION,1024)
