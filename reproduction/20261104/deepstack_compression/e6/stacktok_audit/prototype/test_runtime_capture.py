import unittest
from unittest.mock import patch
import torch
from runtime_capture import VisualCapture,encode_question
class Merger(torch.nn.Module):
 def forward(self,x):
  x.add_(100)
  return x.reshape(-1,4,x.shape[-1]).sum(1)
class Visual(torch.nn.Module):
 def __init__(self):super().__init__();self.spatial_merge_size=2;self.merger=Merger()
 def forward(self,x):
  y=self.merger(x);return y,[y+10,y+20,y+30]
class Tokenizer:
 bos_token_id=1
 def __call__(self,words,**kwargs):
  assert words==['Question:','What?','A.','one'] and kwargs=={'is_split_into_words':True,'return_tensors':'pt','padding':True,'truncation':True}
  return {'input_ids':torch.tensor([[1,2,3]])}
class Tests(unittest.TestCase):
 def test_snapshot_before_mutation_and_hook_cleanup(self):
  v=Visual();old=(tuple(v._forward_hooks),tuple(v.merger._forward_pre_hooks));x=torch.arange(16,dtype=torch.float32).reshape(16,1);before=x.clone()
  with VisualCapture(v,(1,4,4)) as capture:
   out=v(x);r=capture.result();self.assertTrue(torch.equal(r['premerger'],before));self.assertTrue(torch.equal(r['averaged_premerger'],before.reshape(4,4,1).mean(1)));self.assertTrue(torch.equal(r['postmerger'],out[0]));self.assertEqual(len(r['deepstack']),3)
  self.assertEqual(old,(tuple(v._forward_hooks),tuple(v.merger._forward_pre_hooks)))
 def test_second_hook_registration_failure_unwinds_first(self):
  v=Visual();old=(tuple(v._forward_hooks),tuple(v.merger._forward_pre_hooks))
  with patch.object(v,'register_forward_hook',side_effect=RuntimeError('registration failed')):
   with self.assertRaisesRegex(RuntimeError,'registration failed'):
    with VisualCapture(v,(1,4,4)):pass
  self.assertEqual(old,(tuple(v._forward_hooks),tuple(v.merger._forward_pre_hooks)))
 def test_cleanup_on_failure(self):
  v=Visual();old=(tuple(v._forward_hooks),tuple(v.merger._forward_pre_hooks))
  with self.assertRaisesRegex(ValueError,'incomplete'):
   with VisualCapture(v,(1,4,4)) as capture:capture.result()
  self.assertEqual(old,(tuple(v._forward_hooks),tuple(v.merger._forward_pre_hooks)))
 def test_question_tokenization_and_bos(self):
  table=torch.nn.Embedding(8,4);r=encode_question(Tokenizer(),table,[{'type':'text','value':'What?'},{'type':'text','value':'A. one'}],'cpu');self.assertEqual(r['text'],'Question: What? A. one');self.assertEqual(r['input_ids'].tolist(),[[1,2,3]]);self.assertTrue(torch.equal(r['embeddings'],table(torch.tensor([2,3]))))
if __name__=='__main__':unittest.main(verbosity=2)
