import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
D=Path(__file__).resolve().parent;P=D.parent/'pixel_wavefront_model_v2';sys.path.insert(0,str(P))
from test_identity_worker import Model,Selector,fixture
from identity_worker import run_arm
from p3_reconstruct import reconstruct
from offline_identity import validate_pair

CONFIG={'image_token_id':99,'vision_start_token_id':5,'video_token_id':77}
class P3Model(Model):
 def __init__(self):
  super().__init__()
  def rope(ids,grid,video,mask):
   x=reconstruct(ids[0].tolist(),99,5,77,list(range(1024)))
   return torch.tensor(x['position_ids']),torch.tensor(x['rope_delta'])
  self.model.get_rope_index=rope
 def generate(self,**kw):
  Selector()._select_2d_loco_anchored(torch.zeros((4,1)),2,2,'cpu');x=reconstruct(kw['input_ids'][0].tolist(),99,5,77,[0,2]);self.model.rope_deltas=torch.tensor(x['rope_delta'])
  pre={'inputs_embeds':torch.zeros((1,4,2)),'attention_mask':torch.ones((1,4),dtype=torch.long),'position_ids':torch.tensor(x['position_ids']),'visual_pos_masks':torch.tensor([[False,True,True,False]]),'cache_position':torch.arange(4),'deepstack_visual_embeds':[torch.zeros((2,2)) for _ in range(3)]}
  for f in self.model.language_model.pre:f(None,(),pre)
  for f in self.lm_head.post:f(None,(),{},torch.zeros((1,1,4)))
  past=SimpleNamespace(layers=[SimpleNamespace(keys=torch.zeros((1,1,4,2)),values=torch.zeros((1,1,4,2))) for _ in range(36)])
  for f in self.model.language_model.post:f(None,(),{},SimpleNamespace(past_key_values=past))
  return torch.cat((kw['input_ids'],torch.tensor([[2]])),dim=1)
class Tests(unittest.TestCase):
 def test_complete_pair_and_changed_raw_prefix(self):
  with tempfile.TemporaryDirectory() as td:
   inp,kw,expected,anchor=fixture();dirs=[]
   for arm in ('author','wavefront'):
    out=Path(td)/arm;out.mkdir();dirs.append(out);run_arm(torch,P3Model(),inp,Selector,lambda tokens,h,w,t:torch.tensor([0,2]),arm,anchor,expected,kw,out)
   self.assertEqual(validate_pair(*dirs,anchor,expected,CONFIG)['status'],'passed')
   raw=dirs[0]/'returned.json';r=json.loads(raw.read_text());r['full_ids'][0]=6;raw.write_text(json.dumps(r))
   with self.assertRaises(ValueError):validate_pair(*dirs,anchor,expected,CONFIG)
if __name__=='__main__':unittest.main(verbosity=2)
