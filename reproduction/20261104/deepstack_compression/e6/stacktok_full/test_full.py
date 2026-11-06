import sys,unittest
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];sys.path[:0]=[str(HERE),str(BASE/'e6/stacktok_audit/prototype'),str(BASE/'e4')]
from full_adapter import FullStackTokAdapter
from identity_records import tensor_meta
from test_model_adapter import Wrapper,Selector
from full_refs import validate_before_selection,validate_reference
class CountSelector(Selector):
 def __init__(self):self.calls=0
 def stacktok_single(self,*a,**kw):self.calls+=1;return super().stacktok_single(*a,**kw)
class Tests(unittest.TestCase):
 def fixture(self):
  w=Wrapper();inp=w._processor_inputs();x=torch.arange(8192,dtype=torch.float32).reshape(4096,2);main=x.reshape(-1,4,2).mean(1);ds=[main+1,main+2,main+3];pos=torch.arange(1026).view(1,1,-1).expand(3,1,-1);ref={'processor':{k:tensor_meta(inp[k]) for k in inp},'main_visual':tensor_meta(main),'full_deepstack':[tensor_meta(t) for t in ds],'full_position_ids':pos.tolist(),'full_delta':[[3]],'original_length':1026,'image_start':1,'generation_scalars':{},'eos_ids':[3]};return w,x,inp,main,ds,pos,ref
 def test_upstream_call_exactly_once_and_prefill_matches(self):
  w,x,inp,main,ds,pos,ref=self.fixture();sel=CountSelector()
  with FullStackTokAdapter(w,{'message':[{'type':'text','value':'q'}]},sel,256,ref):
   w._processor_inputs();main,ds=w.vision(x);emb=torch.zeros((1,1026,2));emb[:,1:1025]=main
   out=w.lm(inputs_embeds=emb,attention_mask=inp['attention_mask'],position_ids=pos,visual_pos_masks=inp['input_ids']==99,deepstack_visual_embeds=ds,cache_position=torch.arange(1026))
   self.assertEqual(sel.calls,1);self.assertEqual(out['inputs_embeds'].shape[1],258)
 def test_identity_before_select_rejects_mismatch(self):
  w,x,inp,main,ds,pos,ref=self.fixture();self.assertTrue(validate_before_selection(inp,main,ds,pos,torch.tensor([[3]]),1,ref));ref['full_deepstack'][1]={'bad':True}
  with self.assertRaises(ValueError):validate_before_selection(inp,main,ds,pos,torch.tensor([[3]]),1,ref)
 def test_p3_and_three_ds_record_contract(self):
  w,x,inp,main,ds,pos,ref=self.fixture();keep=[0]+list(range(1,257))+[1025];selected=pos[:,:,keep].tolist();r={'processor':ref['processor'],'main_visual':ref['main_visual'],'full_deepstack':ref['full_deepstack'],'rope_delta':[[3]],'generation_scalars':{},'eos_ids':[3],'spatial_ids':list(range(256)),'prefill_position_ids':selected,'decode_trace':[{'position_ids':selected}],'injected_mask':[[False]+[True]*256+[False]],'injected_deepstack':[{}, {}, {}]}
  self.assertTrue(validate_reference(r,ref));r['prefill_position_ids'][2][0][5]+=1
  with self.assertRaises(ValueError):validate_reference(r,ref)
if __name__=='__main__':unittest.main(verbosity=2)
