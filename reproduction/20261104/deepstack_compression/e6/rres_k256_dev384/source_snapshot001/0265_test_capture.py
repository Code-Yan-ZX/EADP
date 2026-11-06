import unittest
import torch
from capture import validate
class CaptureValidation(unittest.TestCase):
 def fixture(self):
  visual=torch.ones((256,2));ds=[visual*(i+2) for i in range(3)];mask=torch.zeros((1,260),dtype=torch.bool);mask[:,2:258]=True;emb=torch.zeros((1,260,2));emb[mask]=visual;pos=torch.arange(260).repeat(3,1).unsqueeze(1)
  return {'block_calls':[1,1], 'merger_calls':1,'ds_calls':[1,1,1], 'visual':visual,'deepstack':ds,'prefill_position_ids':pos,'delta':torch.tensor([[-240]]),'prefill_attention_mask':torch.ones((1,260)),'prefill_visual_pos_masks':mask,'prefill_inputs_embeds':emb,'injected_ds':[x.clone() for x in ds],'logits':torch.ones((1,5))},{'attention_mask':torch.ones((1,260))},pos
 def test_real_shape_and_each_ds(self):
  s,i,p=self.fixture();self.assertTrue(validate(torch,s,i,2,p,[[-240]]))
  for n in range(3):
   s,i,p=self.fixture();s['injected_ds'][n][0,0]+=1
   with self.assertRaises(ValueError):validate(torch,s,i,2,p,[[-240]])
 def test_bad_mask_and_position(self):
  for key in ('prefill_visual_pos_masks','prefill_position_ids'):
   s,i,p=self.fixture();s[key]=s[key].clone();s[key].view(-1)[0]=1
   with self.assertRaises(ValueError):validate(torch,s,i,2,p,[[-240]])
if __name__=='__main__':unittest.main()
