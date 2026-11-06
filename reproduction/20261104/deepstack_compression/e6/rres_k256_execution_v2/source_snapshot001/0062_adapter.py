"""Prepared P3/DS-on adapter for E4 budgets; not approved for GPU timing.

No clones, tensor hashes or device-to-host reads are added. Selector-return,
vision and prefill hooks have nonzero CPU/GPU overhead. A separate high-array
real-model identity diagnostic and independent review remain mandatory.
"""
from contextlib import ExitStack

class BudgetLightAdapter:
 def __init__(self,torch,wrapper,pruner_module,image_start,keep):
  if keep not in (128,256,512,1024) or type(image_start) is not int or image_start<0:raise ValueError('unsupported budget')
  self.t,self.w,self.pruner,self.start,self.keep=torch,wrapper,pruner_module,image_start,keep
  self.state={};self.stack=None
 def __enter__(self):
  if self.stack is not None:raise RuntimeError('nested adapter')
  stack=ExitStack();self.stack=stack;s=self.state
  try:
   original_processor=self.w._processor_inputs;had='_processor_inputs' in self.w.__dict__;stored=self.w.__dict__.get('_processor_inputs')
   def processor(*args,**kwargs):
    x=original_processor(*args,**kwargs);s['inputs']=x;return x
   self.w._processor_inputs=processor
   def restore():
    if had:self.w._processor_inputs=stored
    else:delattr(self.w,'_processor_inputs')
   stack.callback(restore)
   original=self.pruner._greed_select_impl;s['greedy_calls']=0
   def greedy(*args,**kwargs):
    result=original(*args,**kwargs);s['greedy_calls']+=1;s['ordered']=result[0].reshape(-1);return result
   self.pruner._greed_select_impl=greedy;stack.callback(setattr,self.pruner,'_greed_select_impl',original)
   def visual(_module,_args,result):
    if 'deepstack' in s:raise ValueError('unexpected second vision call')
    s['deepstack']=result[1]
   hook=self.w.model.visual.register_forward_hook(visual);stack.callback(hook.remove)
   s['prefill_seen']=False
   def prefill(_module,args,kw):
    if s['prefill_seen']:return None
    s['prefill_seen']=True;x=s['inputs'];t=self.t;k=self.keep;start=self.start
    if k==1024:
     if s['greedy_calls']!=0:raise ValueError('keep-all must bypass greedy')
     spatial=t.arange(1024,device=x['input_ids'].device)
    else:
     if s['greedy_calls']!=1 or s['ordered'].shape[0]!=k:raise ValueError('missing actual author selection')
     spatial=s['ordered'].sort().values
    if 'deepstack' not in s or len(s['deepstack'])!=3:raise ValueError('missing three DS streams')
    full,delta=self.w.model.model.get_rope_index(x['input_ids'],x['image_grid_thw'],None,attention_mask=x['attention_mask'])
    retained=t.cat((t.arange(start,device=spatial.device),start+spatial,t.arange(start+1024,x['input_ids'].shape[1],device=spatial.device)))
    updated=dict(kw);updated['position_ids']=full.index_select(-1,retained);self.w.model.model.rope_deltas=delta+(1024-k)
    mask=t.zeros_like(updated['attention_mask'],dtype=t.bool);mask[:,start:start+k]=True
    updated['visual_pos_masks']=mask;updated['deepstack_visual_embeds']=[v.index_select(0,spatial) for v in s['deepstack']]
    s['selection_mode']='keep_all' if k==1024 else 'author_greedy'
    return args,updated
   hook=self.w.model.model.language_model.register_forward_pre_hook(prefill,with_kwargs=True);stack.callback(hook.remove)
   return self
  except BaseException:stack.close();self.stack=None;raise
 def __exit__(self,typ,value,tb):
  try:return self.stack.__exit__(typ,value,tb)
  finally:self.stack=None
