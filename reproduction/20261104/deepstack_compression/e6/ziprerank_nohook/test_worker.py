import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from worker import hook_counts,run_arm
class Model(torch.nn.Module):
 def __init__(self,fail=False):super().__init__();self.model=SimpleNamespace(rope_deltas='old');self.fail=fail;self.calls=[]
 def set_qi_early_enabled(self,x):self.enabled=x
 def set_qi_early_keep_ratio(self,x):self.ratio=x
 def generate(self,**kwargs):
  self.calls.append(kwargs)
  if self.fail:raise IndexError('synthetic cache failure')
  return SimpleNamespace(sequences=torch.tensor([[1,2]+[7]*33]))
class Processor:tokenizer=SimpleNamespace(encode=lambda text,add_special_tokens:[7])
class Tests(unittest.TestCase):
 def test_no_hooks_and_33_forced(self):
  with tempfile.TemporaryDirectory() as d:
   m=Model();r=run_arm(m,{'input_ids':torch.tensor([[1,2]])},torch,Processor,object,'keep_all',True,1.,Path(d));self.assertEqual(r['status'],'completed');self.assertTrue(r['forced_tokens_match']);self.assertEqual(hook_counts(m),0);self.assertEqual((m.calls[0]['max_new_tokens'],m.calls[0]['min_new_tokens']),(33,33));self.assertFalse(m.calls[0]['do_sample']);self.assertIsNone(m.model.rope_deltas)
 def test_exception_preserved(self):
  with tempfile.TemporaryDirectory() as d:
   r=run_arm(Model(True),{'input_ids':torch.tensor([[1,2]])},torch,Processor,object,'keep_256',True,.25,Path(d));self.assertEqual(r['status'],'failed');self.assertTrue((Path(d)/'keep_256_exception.json').exists())
 def test_forward_hook_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   m=Model();m.register_forward_hook(lambda *_:None)
   with self.assertRaisesRegex(ValueError,'hooks present'):run_arm(m,{'input_ids':torch.tensor([[1,2]])},torch,Processor,object,'disabled',False,1.,Path(d))
if __name__=='__main__':unittest.main(verbosity=2)
