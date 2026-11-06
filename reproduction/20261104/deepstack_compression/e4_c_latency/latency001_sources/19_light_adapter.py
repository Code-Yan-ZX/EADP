"""Low-observer-overhead EADP P3/DS timing adapter; requires separate identity proof.

No hashes, tensor clones, tensor-to-CPU conversion, feature equality checks,
per-layer hooks or explicit synchronization. This does not establish identity.
"""
import sys
from pathlib import Path
from contextlib import ExitStack
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'e4'))
from skip_deepstack import SkipDeepStackMergers

class LightAdapter:
 def __init__(self,torch,wrapper,pruner_module,mode,image_start,keep=256):
  if mode not in ('C-ON','C-DROP','C-SKIP') or keep!=256 or not isinstance(image_start,int) or image_start<0:raise ValueError('unsupported protocol')
  self.t,self.w,self.pruner,self.mode,self.start,self.keep=torch,wrapper,pruner_module,mode,image_start,keep;self.state={};self.stack=None
 def __enter__(self):
  if self.stack is not None:raise RuntimeError('nested entry')
  stack=ExitStack();self.stack=stack
  try:
   if self.mode=='C-SKIP':stack.enter_context(SkipDeepStackMergers(self.w.model.visual,audit_calls=False))
   orig_processor=self.w._processor_inputs;had='_processor_inputs' in self.w.__dict__;stored=self.w.__dict__.get('_processor_inputs')
   def processor(*args,**kwargs):
    x=orig_processor(*args,**kwargs);self.state['inputs']=x;return x
   self.w._processor_inputs=processor
   def restore():
    if had:self.w._processor_inputs=stored
    else:delattr(self.w,'_processor_inputs')
   stack.callback(restore)
   original=self.pruner._greed_select_impl
   def greedy(*args,**kwargs):
    result=original(*args,**kwargs);self.state['ordered']=result[0].reshape(-1);return result
   self.pruner._greed_select_impl=greedy;stack.callback(setattr,self.pruner,'_greed_select_impl',original)
   if self.mode=='C-ON':
    def visual(_module,_args,result):self.state['deepstack']=result[1]
    hook=self.w.model.visual.register_forward_hook(visual);stack.callback(hook.remove)
   self.state['prefill_seen']=False
   def prefill(_module,args,kw):
    if self.state['prefill_seen']:return None
    self.state['prefill_seen']=True;x=self.state['inputs'];spatial=self.state['ordered'].sort().values;start=self.start;t=self.t
    full,delta=self.w.model.model.get_rope_index(x['input_ids'],x['image_grid_thw'],None,attention_mask=x['attention_mask'])
    retained=t.cat((t.arange(start,device=spatial.device),start+spatial,t.arange(start+1024,x['input_ids'].shape[1],device=spatial.device)))
    updated=dict(kw);updated['position_ids']=full.index_select(-1,retained);self.w.model.model.rope_deltas=delta+768
    if self.mode=='C-ON':
     mask=t.zeros_like(updated['attention_mask'],dtype=t.bool);mask[:,start:start+256]=True
     updated['visual_pos_masks']=mask;updated['deepstack_visual_embeds']=[v.index_select(0,spatial) for v in self.state['deepstack']]
    else:updated['visual_pos_masks']=None;updated['deepstack_visual_embeds']=None
    return args,updated
   hook=self.w.model.model.language_model.register_forward_pre_hook(prefill,with_kwargs=True);stack.callback(hook.remove)
   return self
  except BaseException:stack.close();self.stack=None;raise
 def __exit__(self,typ,value,tb):
  try:return self.stack.__exit__(typ,value,tb)
  finally:self.stack=None
