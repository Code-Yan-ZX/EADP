"""Diagnostic-only observation around the actual lightweight timing adapter.

Adds tensor clones and hooks exclusively to identity smoke. This module must
never be loaded into the uninstrumented latency timing path.
"""
from contextlib import ExitStack
from light_adapter import LightAdapter

class LightCapture:
 def __init__(self,torch,wrapper,pruner,mode,image_start):
  self.t=torch;self.w=wrapper;self.pruner=pruner;self.mode=mode;self.start=image_start;self.state={};self.stack=None
 def __enter__(self):
  stack=ExitStack();self.stack=stack
  try:
   adapter=stack.enter_context(LightAdapter(self.t,self.w,self.pruner,self.mode,self.start));self.adapter=adapter;s=self.state
   previous=self.w._processor_inputs;had='_processor_inputs' in self.w.__dict__;stored=self.w.__dict__.get('_processor_inputs')
   def processor(*args,**kwargs):
    value=previous(*args,**kwargs);s['inputs']={k:(x.detach().clone() if hasattr(x,'detach') else x) for k,x in value.items()};return value
   self.w._processor_inputs=processor
   def restore():
    if had:self.w._processor_inputs=stored
    else:delattr(self.w,'_processor_inputs')
   stack.callback(restore)
   vision=self.w.model.visual;s['block_calls']=[0]*len(vision.blocks);s['main_merger_calls']=0;s['ds_calls']=[0,0,0];s['ds_none']=[None,None,None]
   def block_hook(i):
    def count(_m,_args,_out):s['block_calls'][i]+=1
    return count
   for i,block in enumerate(vision.blocks):
    h=block.register_forward_hook(block_hook(i));stack.callback(h.remove)
   def main_count(_m,_args,_out):s['main_merger_calls']+=1
   h=vision.merger.register_forward_hook(main_count);stack.callback(h.remove)
   def ds_hook(i):
    def count(_m,_args,out):s['ds_calls'][i]+=1;s['ds_none'][i]=out is None
    return count
   for i,module in enumerate(vision.deepstack_merger_list):
    h=module.register_forward_hook(ds_hook(i));stack.callback(h.remove)
   def visual(_m,_args,out):
    s['visual']=out[0].detach().clone()
    if out[1] is None or len(out[1])!=3:raise ValueError('expected three DS slots')
    absent=[x is None for x in out[1]]
    if absent!=([True]*3 if self.mode=='C-SKIP' else [False]*3):raise ValueError('DS merger output mismatch')
    s['deepstack']=None if self.mode=='C-SKIP' else [x.detach().clone() for x in out[1]]
   h=self.w.model.visual.register_forward_hook(visual);stack.callback(h.remove)
   def pruned(_m,_args,out):
    s['pruned']=out[0].detach().clone();s['ordered']=adapter.state['ordered'].detach().clone()
   h=self.w.pruner.register_forward_hook(pruned);stack.callback(h.remove)
   def prefill(_m,_args,kw):
    if 'prefill_embeds' in s:return
    for key in ('inputs_embeds','attention_mask','position_ids','visual_pos_masks'):
     value=kw.get(key);s['prefill_'+key]=None if value is None else value.detach().clone()
    ds=kw.get('deepstack_visual_embeds');s['injected_ds']=None if ds is None else [x.detach().clone() for x in ds]
    s['adjusted_delta']=self.w.model.model.rope_deltas.detach().clone()
   h=self.w.model.model.language_model.register_forward_pre_hook(prefill,with_kwargs=True);stack.callback(h.remove)
   def logits(_m,_args,out):
    if 'logits' not in s:s['logits']=out[:,-1,:].detach().clone()
   h=self.w.model.lm_head.register_forward_hook(logits);stack.callback(h.remove)
   return self
  except BaseException:stack.close();self.stack=None;raise
 def __exit__(self,typ,value,tb):
  try:return self.stack.__exit__(typ,value,tb)
  finally:self.stack=None
