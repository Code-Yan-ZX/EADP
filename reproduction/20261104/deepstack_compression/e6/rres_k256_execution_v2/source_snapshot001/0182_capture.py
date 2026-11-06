"""Diagnostic hooks for the stock 512 processor path; not a timing tool."""
from contextlib import ExitStack

class Capture:
 def __init__(self,torch,wrapper):self.t=torch;self.w=wrapper;self.state={}
 def __enter__(self):
  s=self.state;stack=ExitStack();self.stack=stack
  try:
   vision=self.w.model.visual;language=self.w.model.model.language_model
   s['block_calls']=[0]*len(vision.blocks);s['merger_calls']=0;s['ds_calls']=[0,0,0]
   def block(i):
    def hook(_m,_a,_out):s['block_calls'][i]+=1
    return hook
   for i,m in enumerate(vision.blocks):stack.callback(m.register_forward_hook(block(i)).remove)
   def merger(_m,_a,_out):s['merger_calls']+=1
   stack.callback(vision.merger.register_forward_hook(merger).remove)
   def ds(i):
    def hook(_m,_a,out):
     if out is None:raise ValueError('empty DeepStack merger')
     s['ds_calls'][i]+=1
    return hook
   if len(vision.deepstack_merger_list)!=3:raise ValueError('DeepStack merger count')
   for i,m in enumerate(vision.deepstack_merger_list):stack.callback(m.register_forward_hook(ds(i)).remove)
   def visual(_m,_a,out):
    if 'visual' in s or len(out)!=2 or len(out[1])!=3:raise ValueError('visual output')
    s['visual']=out[0].detach().clone();s['deepstack']=[x.detach().clone() for x in out[1]]
   stack.callback(vision.register_forward_hook(visual).remove)
   def prefill(_m,_a,kw):
    if 'prefill_inputs_embeds' in s:return
    for name in ('inputs_embeds','attention_mask','position_ids','visual_pos_masks'):
     value=kw.get(name);s['prefill_'+name]=None if value is None else value.detach().clone()
    ds=kw.get('deepstack_visual_embeds');s['injected_ds']=None if ds is None else [x.detach().clone() for x in ds]
    delta=self.w.model.model.rope_deltas;s['delta']=None if delta is None else delta.detach().clone()
   stack.callback(language.register_forward_pre_hook(prefill,with_kwargs=True).remove)
   def logits(_m,_a,out):
    if 'logits' not in s:s['logits']=out[:,-1,:].detach().clone()
   stack.callback(self.w.model.lm_head.register_forward_hook(logits).remove)
   return self
  except BaseException:stack.close();raise
 def __exit__(self,*args):return self.stack.__exit__(*args)

def validate(torch,state,inputs,start,position_ids,delta):
 s=state
 if s['block_calls']!=[1]*len(s['block_calls']) or s['merger_calls']!=1 or s['ds_calls']!=[1]*3:raise ValueError('visual/merger call counts')
 if s['visual'].shape[0]!=256 or len(s['deepstack'])!=3 or any(x.shape[0]!=256 for x in s['deepstack']):raise ValueError('visual and DS lengths')
 if not torch.equal(s['prefill_position_ids'],position_ids) or s['delta'].tolist()!=delta:raise ValueError('position or delta')
 if not torch.equal(s['prefill_attention_mask'],inputs['attention_mask']):raise ValueError('prefill attention mask')
 mask=s['prefill_visual_pos_masks']
 if mask is None or mask.ndim!=2 or int(mask.sum())!=256:raise ValueError('visual mask')
 expected=torch.zeros_like(mask,dtype=torch.bool);expected[:,start:start+256]=True
 if not torch.equal(mask.bool(),expected):raise ValueError('visual span mask')
 if not torch.equal(s['prefill_inputs_embeds'][mask],s['visual']):raise ValueError('visual input embedding')
 if s['injected_ds'] is None or len(s['injected_ds'])!=3 or any(not torch.equal(a,b) for a,b in zip(s['deepstack'],s['injected_ds'])):raise ValueError('DeepStack injection')
 if not torch.isfinite(s['logits']).all():raise ValueError('nonfinite logits')
 return True
