"""CPU synthetic hook wiring through both actual diagnostic run_arm functions.

Model/adapter/observer are fakes; this checks real return schemas and admission
ordering, not the actual model's tensor capture correctness.
"""
import copy,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
import torch
BASE=Path(__file__).resolve().parent.parent
for path in (BASE/'e4',BASE/'e4_c_budget_pilot_v2',BASE/'e4_c_multibudget_fixed32',BASE/'e4_c_multibudget_references_v2'):sys.path.insert(0,str(path))
import reference_arm as fresh_fixed,light_fixed,identity_records
import importlib.util
spec=importlib.util.spec_from_file_location("_bulk_probe_core",BASE/"e4_c_multibudget_references_v2/core.py");bulk=importlib.util.module_from_spec(spec);spec.loader.exec_module(bulk)
spec=importlib.util.spec_from_file_location("_fixed_probe_core",BASE/"e4_c_multibudget_fixed32/core.py");fixed=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixed)

class Handle:
 def __init__(self,remove):self.remove=remove
class Hookable:
 def __init__(self):self.hooks=[]
 def register_forward_hook(self,fn,**kwargs):
  self.hooks.append(fn);return Handle(lambda:self.hooks.remove(fn))
class FakeKV:
 def __init__(self,n):self.shape=[1,8,n,128]
class FakeObserver:
 def __init__(self,w):self.trace=w.trace
 def __enter__(self):return self
 def __exit__(self,*args):pass
class FreshAdapter:
 state_value=None
 def __init__(self,*args):self.state=copy.copy(self.state_value)
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def verify_calls(self):pass
class LightAdapter:
 state_value=None
 def __init__(self,*args):self.state=copy.copy(self.state_value);self.adapter=types.SimpleNamespace(state={'greedy_calls':1})
 def __enter__(self):return self
 def __exit__(self,*args):pass
class Tests(unittest.TestCase):
 def test_actual_arm_return_schema_and_fingerprint_flow(self):
  k=128;n=1028;start=2;seq=n-1024+k;delta=1024-k
  ordered=torch.arange(k);retained=torch.cat((torch.arange(start),start+ordered,torch.arange(start+1024,n)))
  full=torch.arange(n).reshape(1,1,n).repeat(3,1,1);pos=full.index_select(2,retained)
  vis=torch.arange(1024*4,dtype=torch.float32).reshape(1024,4);pruned=vis[:k];pre=torch.zeros(1,seq,4);pre[0,start:start+k]=pruned
  ds=[vis.clone() for _ in range(3)];injected=[v[:k] for v in ds];inputs={'input_ids':torch.arange(n).reshape(1,n),'attention_mask':torch.ones(1,n),'pixel_values':torch.ones(1,4),'image_grid_thw':torch.tensor([[1,64,64]])}
  trace=[]
  for i in range(32):
   prev=0 if i==0 else seq+i-1;after=seq+i
   trace.append({'position_ids':pos.tolist() if i==0 else [[[seq+i-1+delta]]]*3,'cache_position':list(range(seq)) if i==0 else [seq+i-1],'rope_delta':[[delta]],'layer_cache_lengths':[prev]*36,'layer_cache_lengths_after':[after]*36})
  common={'inputs':inputs,'visual':vis,'pruned':pruned,'deepstack':ds,'ordered':ordered,'spatial':ordered,'retained':retained,'full_pos':full,'full_delta':torch.tensor([[0]]),'prefill_position_ids':pos,'adjusted_delta':torch.tensor([[delta]]),'prefill_embeds':pre,'prefill_mask':torch.ones(1,seq),'prefill_attention_mask':torch.ones(1,seq),'actual_ds_mask':torch.tensor([[False]*start+[True]*k+[False]*2]),'actual_deepstack':injected,'selected_deepstack':injected,'start':start,'greedy_calls':1,'selection_mode':'author_greedy','actual_injection_verified':True,'visual_calls':1,'main_merger_calls':1,'block_calls':[1],'ds_calls':[1,1,1],'ds_none':[False]*3,'prefill_inputs_embeds':pre,'prefill_visual_pos_masks':torch.tensor([[False]*start+[True]*k+[False]*2]),'injected_ds':injected,'logits':torch.ones(1,4)}
  FreshAdapter.state_value=common;LightAdapter.state_value=common
  language=Hookable();head=Hookable()
  class Model:
   generation_config=types.SimpleNamespace(eos_token_id=9)
   def __init__(self):
    self.model=types.SimpleNamespace(language_model=language,get_rope_index=lambda *_a,**_kw:(full,torch.tensor([[0]])))
    self.visual=types.SimpleNamespace(blocks=[object()]);self.lm_head=head
   def generate(self,**kwargs):
    for h in list(head.hooks):h(None,None,torch.ones(1,1,4))
    for i in range(32):
     layers=[types.SimpleNamespace(keys=FakeKV(seq+i),values=FakeKV(seq+i)) for _ in range(36)]
     out=types.SimpleNamespace(past_key_values=types.SimpleNamespace(layers=layers))
     for h in list(language.hooks):h(None,None,None,out)
    return torch.tensor([list(range(100,132))])
  w=types.SimpleNamespace(model=Model(),visual_token_num=k,pruner=types.SimpleNamespace(visual_token_num=k),trace=trace)
  row={'task':'D','qid':'1','group_id':'g','image_sha256':'i','message_sha256':'m','image_start':start,'split':'DEV'}
  recipe={'do_sample':False,'max_new_tokens':2048,'top_p':.001,'top_k':1,'temperature':.01,'repetition_penalty':1.0};panel={'generation_scalars':recipe,'eos_ids':[9]}
  def generate(w,_row):w.model.generate(**recipe);return 'answer'
  def archive(tensors):return {'kind':'summary_only','tensors':{}}
  saved=[];real_meta=identity_records.tensor_meta
  def meta(t):return {'dtype':'torch.bfloat16','shape':t.shape,'sha256':'a'*64} if isinstance(t,FakeKV) else real_meta(t)
  decode=types.SimpleNamespace(DecodeObserver=FakeObserver,validate_trace=lambda *_:None)
  with patch.object(fresh_fixed,'BudgetAdapter',FreshAdapter),patch.object(fresh_fixed,'DecodeObserver',FakeObserver),patch.object(fresh_fixed,'validate_trace',lambda *_:None),patch.object(light_fixed,'BudgetCapture',LightAdapter),patch.object(light_fixed,'load_decode',lambda:decode),patch.object(identity_records,'tensor_meta',meta),patch.object(fresh_fixed,'tensor_meta',meta),patch.object(light_fixed,'tensor_meta',meta):
   fresh=fresh_fixed.run_arm(torch,w,None,row,k,True,generate,saved.append,archive,capture_prefill_logits=True,accepted_panel=panel)
   light=light_fixed.run_arm(torch,w,None,row,k,generate,saved.append,archive,accepted_panel=panel)
  self.assertEqual(len(saved),2);self.assertEqual(len(fresh['cache_fingerprints']),1);self.assertEqual(len(light['cache_fingerprints']),32)
  panel_record=copy.deepcopy(fresh);panel_record['actual_injection_verified']=True;panel_record['generation_scalars']=recipe
  self.assertTrue(bulk.validate(fresh,panel_record,fixed))
  broken=copy.deepcopy(fresh);del broken['cache_fingerprints']
  with self.assertRaisesRegex(ValueError,'fingerprint'):bulk.validate(broken,panel_record,fixed)
  self.assertNotIn('generate',vars(w.model));self.assertEqual(language.hooks,[])
if __name__=='__main__':unittest.main()
