import unittest
from types import SimpleNamespace
import torch
from adapter import BudgetLightAdapter
class Vision(torch.nn.Module):
 def forward(self):
  x=torch.arange(1024,dtype=torch.float32).reshape(1024,1)
  return x,[x+1000,x+2000,x+3000]
class LM(torch.nn.Module):
 def forward(self,**kw):return kw
class Core:
 def __init__(self):self.language_model=LM();self.rope_deltas=None
 def get_rope_index(self,ids,grid,video,attention_mask=None):
  n=ids.shape[1];p=torch.arange(n).reshape(1,1,n).repeat(3,1,1)
  return p,torch.tensor([[11]])
class Wrapper:
 def __init__(self):self.model=SimpleNamespace(visual=Vision(),model=Core())
 def _processor_inputs(self,*args,**kw):return {'input_ids':torch.arange(1028).reshape(1,-1),'attention_mask':torch.ones((1,1028),dtype=torch.long),'image_grid_thw':torch.tensor([[1,64,64]])}
class Tests(unittest.TestCase):
 def test_all_budgets_real_hooks_p3_ds_decode_and_cleanup(self):
  for k in (128,256,512,1024):
   with self.subTest(k=k):
    w=Wrapper();original=lambda n:(torch.arange(n-1,-1,-1).reshape(1,-1),torch.tensor([n]));native=SimpleNamespace(_greed_select_impl=original)
    with BudgetLightAdapter(torch,w,native,2,k) as a:
     w._processor_inputs();w.model.visual()
     if k<1024:native._greed_select_impl(k)
     result=w.model.model.language_model(attention_mask=torch.ones((1,k+4),dtype=torch.long))
     pos=torch.cat((torch.arange(2),2+torch.arange(k),torch.arange(1026,1028)))
     self.assertTrue(torch.equal(result['position_ids'],pos.reshape(1,1,-1).repeat(3,1,1)))
     self.assertEqual(w.model.model.rope_deltas.tolist(),[[11+1024-k]])
     self.assertEqual(int(result['visual_pos_masks'].sum()),k)
     for n,ds in enumerate(result['deepstack_visual_embeds']):self.assertTrue(torch.equal(ds[:,0],torch.arange(k,dtype=torch.float32)+(n+1)*1000))
     self.assertEqual(a.state['greedy_calls'],0 if k==1024 else 1)
     decode=w.model.model.language_model(attention_mask=torch.ones((1,k+5),dtype=torch.long));self.assertNotIn('position_ids',decode)
    self.assertIs(native._greed_select_impl,original);self.assertEqual(len(w.model.visual._forward_hooks),0);self.assertEqual(len(w.model.model.language_model._forward_pre_hooks),0)
 def test_missing_greedy_or_ds_rejected(self):
  w=Wrapper();native=SimpleNamespace(_greed_select_impl=lambda n:(torch.arange(n).reshape(1,-1),None))
  with self.assertRaisesRegex(ValueError,'missing actual author'):
   with BudgetLightAdapter(torch,w,native,2,128):w._processor_inputs();w.model.visual();w.model.model.language_model(attention_mask=torch.ones((1,132)))
  self.assertEqual(len(w.model.visual._forward_hooks),0)
 def test_keep_all_rejects_any_greedy_and_invalid_budget(self):
  w=Wrapper();native=SimpleNamespace(_greed_select_impl=lambda n:(torch.arange(n).reshape(1,-1),None))
  with self.assertRaisesRegex(ValueError,'bypass'):
   with BudgetLightAdapter(torch,w,native,2,1024):w._processor_inputs();w.model.visual();native._greed_select_impl(1024);w.model.model.language_model(attention_mask=torch.ones((1,1028)))
  with self.assertRaises(ValueError):BudgetLightAdapter(torch,w,native,2,129)
if __name__=='__main__':unittest.main(verbosity=2)
