import unittest
from types import SimpleNamespace
import torch
from capture import CaptureInputs

class Merger(torch.nn.Module):
 def __init__(self,offset):super().__init__();self.offset=offset
 def forward(self,x):return x+self.offset
class Vision(torch.nn.Module):
 def __init__(self):super().__init__();self.deepstack_merger_list=torch.nn.ModuleList([Merger(1),Merger(2),Merger(3)])
 def forward(self,x):return x,[m(x) for m in self.deepstack_merger_list]
class Pruner(torch.nn.Module):
 def forward(self,x):return x[:256],[256]
class LM(torch.nn.Module):
 def forward(self,attention_mask=None,visual_pos_masks=None,deepstack_visual_embeds=None):return attention_mask

class CaptureTests(unittest.TestCase):
 def prepare(self):
  vision=Vision();lm=LM();pruner=Pruner();w=SimpleNamespace(pruner=pruner,model=SimpleNamespace(visual=vision,model=SimpleNamespace(language_model=lm)))
  adapter=SimpleNamespace(state={'ordered':torch.arange(255,-1,-1)});cap=CaptureInputs(torch,w,adapter);x=torch.arange(2048).reshape(1024,2)
  with cap:
   _visual,ds=vision(x);pruner(x);mask=torch.zeros((1,258),dtype=torch.bool);mask[:,1:257]=True
   lm(attention_mask=torch.ones((1,258)),visual_pos_masks=mask,deepstack_visual_embeds=[d[:256] for d in ds])
  return cap,x,vision,lm,pruner
 def test_capture_replays_full_selector_three_mergers_and_injection(self):
  cap,x,vision,lm,pruner=self.prepare();x.fill_(-999);operations=cap.prepare_operations(1)
  self.assertEqual(set(operations),{'selector','ds_merger_three_streams','ds_injection'})
  self.assertEqual(operations['selector']()[1],[256]);self.assertEqual(len(operations['ds_merger_three_streams']()),3)
  self.assertEqual(int(operations['ds_injection']()[0].sum()),256)
  self.assertFalse(vision._forward_hooks);self.assertFalse(lm._forward_pre_hooks);self.assertFalse(pruner._forward_hooks)
 def test_mismatched_injection_is_rejected(self):
  cap,*_=self.prepare();cap.state['injection']['streams'][0][0,0]+=1
  with self.assertRaisesRegex(ValueError,'injection replay differs'):cap.prepare_operations(1)
if __name__=='__main__':unittest.main()
