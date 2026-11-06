"""Diagnostic P3/DeepStack adapter for local Div/CD selection outputs.

This never replaces the selection algorithm. Exact bitwise row identity must
uniquely recover every selected spatial index; ambiguous images fail closed.
Performance measurement and full experiment admission are out of scope.
"""
from pathlib import Path
import sys,types
HERE=Path(__file__).resolve().parent;E4=HERE.parents[1]/'e4';sys.path.insert(0,str(E4))
from budget_adapter import BudgetAdapter
from recover_indices import recover

class SelectedBaselineAdapter(BudgetAdapter):
 def __init__(self,torch,wrapper,method):
  if method not in ('div','cd'):raise ValueError('unsupported baseline')
  expected={'div':'DivPruner','cd':'CDPruner'}[method]
  if type(wrapper.pruner).__name__!=expected:raise ValueError('not the actual requested author pruner')
  def forbidden(*_args,**_kwargs):raise RuntimeError('EADP greedy must not run')
  super().__init__(torch,wrapper,types.SimpleNamespace(_greed_select_impl=forbidden),256,True)
  self.method=method;self.state['selector_calls']=0
 def __enter__(self):
  super().__enter__()
  try:
   def observe(*_):self.state['selector_calls']+=1
   h=self.w.pruner.register_forward_hook(observe);self.stack.callback(h.remove);return self
  except BaseException:
   self.stack.close();self.stack=None;raise
 def _prefill(self,_module,args,kw):
  s,t=self.state,self.torch;s['llm_calls']+=1
  if s['llm_calls']!=1:return None
  if s['visual_calls']!=1 or s['selector_calls']!=1 or s['greedy_calls']!=0 or s['splits']!=[256]:raise ValueError('actual visual/selector calls')
  ids=recover(t,s['visual'],s['pruned'])
  if len(ids)!=256 or any(i<0 or i>=1024 for i in ids):raise ValueError('actual selected IDs')
  spatial=t.tensor(ids,dtype=t.long,device=s['visual'].device);x=s['inputs'];start=s['start'];length=x['input_ids'].shape[1]
  full,delta=self.w.model.model.get_rope_index(x['input_ids'],x['image_grid_thw'],None,attention_mask=x['attention_mask'])
  retained=t.cat((t.arange(start,device=spatial.device),start+spatial,t.arange(start+1024,length,device=spatial.device)))
  embeds=self.w.model.get_input_embeddings()(x['input_ids']);expected=t.cat((embeds[:,:start],s['pruned'].unsqueeze(0),embeds[:,start+1024:]),dim=1)
  if not t.equal(expected,kw['inputs_embeds']) or not t.equal(x['attention_mask'].index_select(1,retained),kw['attention_mask']):raise ValueError('actual prefill differs')
  streams=s['deepstack']
  if len(streams)!=3 or any(v.shape!=s['visual'].shape for v in streams):raise ValueError('unaligned full DeepStack')
  selected=[v.index_select(0,spatial) for v in streams];mask=t.zeros_like(kw['attention_mask'],dtype=t.bool);mask[:,start:start+256]=True
  kw=dict(kw);kw['position_ids']=full.index_select(-1,retained);kw['visual_pos_masks']=mask;kw['deepstack_visual_embeds']=selected;self.w.model.model.rope_deltas=delta+768
  s.update(ordered=spatial.clone(),spatial=spatial,retained=retained,full_pos=full,full_delta=delta,selected_deepstack=selected,intended_ds_mask=mask,prefill_position_ids=kw['position_ids'],prefill_embeds=kw['inputs_embeds'],prefill_mask=kw['attention_mask'],adjusted_delta=self.w.model.model.rope_deltas,selection_mode='author_'+self.method)
  return args,kw
 def verify_calls(self):
  s=self.state
  if (s['visual_calls']!=1 or s['selector_calls']!=1 or s['greedy_calls']!=0 or not s['llm_calls'] or s['main_merger_calls']!=1 or not s.get('actual_injection_verified') or any(n!=1 for n in s['block_calls']) or s['ds_calls']!=[1,1,1] or sorted(s['ds_results'])!=[(0,False),(1,False),(2,False)]):raise ValueError('actual baseline path incomplete')
  return True
