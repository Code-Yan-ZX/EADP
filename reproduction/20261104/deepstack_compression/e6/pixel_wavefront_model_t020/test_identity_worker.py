import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from identity_worker import run_arm,tensor_meta
from identity_contract import compare_pair
from archive_contract import validate_archive
class Hook:
 def __init__(self,owner,item):self.owner=owner;self.item=item
 def remove(self):self.owner.remove(self.item)
class Module:
 def __init__(self):self.pre=[];self.post=[]
 def register_forward_pre_hook(self,fn,with_kwargs=False):self.pre.append(fn);return Hook(self.pre,fn)
 def register_forward_hook(self,fn,with_kwargs=False):self.post.append(fn);return Hook(self.post,fn)
class Selector:
 method='max';threshold=.2;anchored=True
 def _select_2d_loco_anchored(self,tokens,h,w,device):return torch.tensor([0,2])
class Model:
 def __init__(self):
  self.model=SimpleNamespace(language_model=Module(),rope_deltas=None,get_rope_index=lambda ids,grid,v,mask:(torch.arange(ids.shape[1]).view(1,1,-1).expand(3,1,-1),torch.tensor([[0]])));self.lm_head=Module();self.config=SimpleNamespace(image_token_id=99);self.generation_config=SimpleNamespace(eos_token_id=2)
 def generate(self,**kw):
  Selector()._select_2d_loco_anchored(torch.zeros((4,1)),2,2,'cpu');self.model.rope_deltas=torch.tensor([[0]]);selected=torch.tensor([0,1,3,1025]);pos=torch.arange(1026).view(1,1,-1).expand(3,1,-1).index_select(2,selected)
  pre={'inputs_embeds':torch.zeros((1,4,2)),'attention_mask':torch.ones((1,4),dtype=torch.long),'position_ids':pos,'visual_pos_masks':torch.tensor([[False,True,True,False]]),'cache_position':torch.arange(4),'deepstack_visual_embeds':[torch.zeros((2,2)) for _ in range(3)]}
  for f in self.model.language_model.pre:f(None,(),pre)
  for f in self.lm_head.post:f(None,(),{},torch.zeros((1,1,4)))
  past=SimpleNamespace(layers=[SimpleNamespace(keys=torch.zeros((1,1,4,2)),values=torch.zeros((1,1,4,2))) for _ in range(36)])
  for f in self.model.language_model.post:f(None,(),{},SimpleNamespace(past_key_values=past))
  return torch.cat((kw['input_ids'],torch.tensor([[2]])),dim=1)
def fixture():
 inp={'input_ids':torch.tensor([[5]+[99]*1024+[6]]),'attention_mask':torch.ones((1,1026),dtype=torch.long),'pixel_values':torch.zeros((4096,2)),'image_grid_thw':torch.tensor([[1,64,64]])};kwargs={'do_sample':False,'max_new_tokens':2048};expected={'selected_indices':[0,2],'tokens':[2],'eos_ids':[2],'processor':{k:tensor_meta(torch,v) for k,v in inp.items()},'generation_kwargs':dict(kwargs)};anchor={'task':'x','qid':'1','group_id':'g','message_sha256':'m','image_sha256':'i'};return inp,kwargs,expected,anchor
class Tests(unittest.TestCase):
 def test_actual_worker_records_feed_contract(self):
  with tempfile.TemporaryDirectory() as td:
   inp,kw,expected,anchor=fixture();results=[]
   for arm in ('author','wavefront'):
    out=Path(td)/arm;out.mkdir();results.append(run_arm(torch,Model(),inp,Selector,lambda tokens,h,w,t:torch.tensor([0,2]),arm,anchor,expected,kw,out))
   self.assertTrue(compare_pair(*results,anchor,kw));self.assertTrue(all(validate_archive(r) for r in results))
   p=Path(results[0]['archive_path']);p.write_bytes(p.read_bytes()+b'tamper')
   with self.assertRaises(ValueError):validate_archive(results[0])
 def test_success_both_and_restore(self):
  for arm in ('author','wavefront'):
   with self.subTest(arm=arm),tempfile.TemporaryDirectory() as td:
    model=Model();old=Selector._select_2d_loco_anchored;inp,kw,expected,anchor=fixture();r=run_arm(torch,model,inp,Selector,lambda tokens,h,w,t:torch.tensor([0,2]),arm,anchor,expected,kw,Path(td));self.assertEqual(r['output_ids'],[2]);self.assertEqual(len(r['cache_fingerprints'][0]),36);self.assertIs(Selector._select_2d_loco_anchored,old);self.assertEqual(model.model.language_model.pre,[]);self.assertEqual(model.model.language_model.post,[])
 def test_returned_is_durable_when_identity_fails(self):
  with tempfile.TemporaryDirectory() as td:
   model=Model();old=Selector._select_2d_loco_anchored;inp,kw,expected,anchor=fixture();expected['selected_indices']=[0,1]
   with self.assertRaises(ValueError):run_arm(torch,model,inp,Selector,None,'author',anchor,expected,kw,Path(td))
   raw=json.loads((Path(td)/'returned.json').read_text());self.assertEqual(raw['full_ids'][-1],2);self.assertEqual(raw['input_length'],1026);events=[json.loads(x)['event'] for x in (Path(td)/'events.jsonl').read_text().splitlines()];self.assertEqual(events,['generation_started','selector_returned','generation_returned']);self.assertIs(Selector._select_2d_loco_anchored,old);self.assertEqual(model.model.language_model.pre,[])
 def test_wrong_recipe_or_processor_before_generate(self):
  for key in ('recipe','processor'):
   with tempfile.TemporaryDirectory() as td:
    model=Model();inp,kw,expected,anchor=fixture()
    if key=='recipe':kw['top_k']=2
    else:inp['input_ids'][0,0]=7
    with self.assertRaises(ValueError):run_arm(torch,model,inp,Selector,None,'author',anchor,expected,kw,Path(td))
    self.assertFalse((Path(td)/'returned.json').exists())
if __name__=='__main__':unittest.main(verbosity=2)
