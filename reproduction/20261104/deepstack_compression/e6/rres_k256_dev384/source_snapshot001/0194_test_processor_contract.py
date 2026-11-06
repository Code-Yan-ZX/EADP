import copy,unittest
from types import SimpleNamespace
import torch
from processor_contract import check
class TestProcessor(unittest.TestCase):
 def setUp(self):
  self.cfg=SimpleNamespace(image_token_id=99,vision_config=SimpleNamespace(spatial_merge_size=2));self.ref={'processor':{},'image_start':2}
  def inp(n,grid):return {'input_ids':torch.tensor([[1,2]+[99]*n+[3,4]]),'attention_mask':torch.ones((1,n+4),dtype=torch.long),'pixel_values':torch.ones(3,2),'image_grid_thw':torch.tensor([grid])}
  self.original=inp(1024,[1,64,64]);self.low=inp(256,[1,32,32])
 def rope(self,_):
  prefix=torch.arange(2).view(1,-1).expand(3,-1);h=torch.arange(16).view(-1,1).expand(16,16).reshape(-1);w=torch.arange(16).view(1,-1).expand(16,16).reshape(-1);visual=torch.stack((torch.zeros(256,dtype=torch.long),h,w))+2;suffix=torch.arange(18,20).view(1,-1).expand(3,-1);p=torch.cat((prefix,visual,suffix),1).unsqueeze(1);return p,torch.tensor([[int(p.max()+1-260)]])
 def runcheck(self):return check(torch,self.original,self.low,self.cfg,self.ref,lambda *args:None,self.rope)
 def test_good(self):self.assertEqual(self.runcheck()['low_visual_tokens'],256)
 def test_bad_grid_and_text(self):
  self.low['image_grid_thw'][0,1]=31
  with self.assertRaises(ValueError):self.runcheck()
  self.low['image_grid_thw'][0,1]=32;self.low['input_ids'][0,-1]=5
  with self.assertRaises(ValueError):self.runcheck()
 def test_bad_visual_or_position(self):
  self.low['input_ids'][0,3]=5
  with self.assertRaises(ValueError):self.runcheck()
  self.low['input_ids'][0,3]=99;self.rope=lambda _: (torch.zeros((3,1,260),dtype=torch.long),torch.tensor([[0]]))
  with self.assertRaises(ValueError):self.runcheck()
if __name__=='__main__':unittest.main()
