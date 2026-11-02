"""Synthetic real PyTorch pre-hook checks on CPU only; no model or labels."""
import json,os,sys
import torch
from pathlib import Path
import common as c
from intervention import change
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
if '--author' in sys.argv:assert not any(Path('/dev',n).exists() for n in ('nvidia0','nvidiactl','nvidia-uvm'))
class FakeLM(torch.nn.Module):
 def forward(self, inputs_embeds, position_ids, cache_position, visual_pos_masks, deepstack_visual_embeds):
  result=inputs_embeds.clone()
  if visual_pos_masks is not None and deepstack_visual_embeds is not None:
   for layer in deepstack_visual_embeds:result[visual_pos_masks]+=layer
  return result

def case(arm,original_on,allow_mask=True):
 lm=FakeLM();counts={'calls':0,'applied':False,'error':None,'continuation_untouched':False,'final_observed':None}
 em=torch.zeros(1,4,3);pos=torch.arange(12).view(3,1,4);cache=torch.arange(4)
 mask=torch.tensor([[False,True,True,False]])
 ds=[torch.full((2,3),float(x)) for x in (1,2,3)]
 def hook(module,args,kwargs):
  counts['calls']+=1
  if counts['calls']==1:
   changed,applied,error=change(arm,kwargs,mask if allow_mask else None,ds)
   counts.update(applied=applied,error=error)
   for k in ('inputs_embeds','position_ids','cache_position'):assert changed[k] is kwargs[k]
   if changed is not kwargs:return args,changed
  else:counts['continuation_untouched']=kwargs['visual_pos_masks'] is None and kwargs['deepstack_visual_embeds'] is None
 def final_observer(module,args,kwargs):
  if counts['calls']==1:counts['final_observed']=(kwargs['visual_pos_masks'],kwargs['deepstack_visual_embeds'])
  return None
 handle=lm.register_forward_pre_hook(hook,with_kwargs=True)
 second=lm.register_forward_pre_hook(final_observer,with_kwargs=True)
 try:
  out=lm(inputs_embeds=em,position_ids=pos,cache_position=cache,
      visual_pos_masks=mask if original_on else None,deepstack_visual_embeds=ds if original_on else None)
  follow=lm(inputs_embeds=em,position_ids=pos,cache_position=cache,visual_pos_masks=None,deepstack_visual_embeds=None)
 finally:handle.remove();second.remove()
 assert counts['calls']==2 and counts['continuation_untouched'] and torch.equal(follow,em)
 if arm=='plain_ds_off':assert counts['final_observed']==(None,None)
 if arm=='eadp1024_ds_on' and allow_mask:assert counts['final_observed'][0] is mask and counts['final_observed'][1] is ds
 return out,counts
base_on,plain=case('plain',True);off,plainoff=case('plain_ds_off',True)
base_off,eadp=case('eadp1024',False);on,eadpon=case('eadp1024_ds_on',False)
invalid,bad=case('eadp1024_ds_on',False,allow_mask=False)
assert torch.equal(base_on,on) and torch.equal(base_off,off) and torch.equal(invalid,base_off)
assert plainoff['applied'] and eadpon['applied'] and not plain['applied'] and not eadp['applied']
assert not bad['applied'] and bad['error'] and not torch.cuda.is_initialized()
raw=dict(inputs_embeds=torch.zeros(1,4,3),visual_pos_masks=None,deepstack_visual_embeds=None)
mask=torch.tensor([[False,True,True,False]])
for ds in ([torch.zeros(1,3)]*3,[torch.zeros(2,3,dtype=torch.float64)]*3):
 changed,applied,error=change('eadp1024_ds_on',raw,mask,ds)
 assert changed is raw and not applied and error
report=dict(success=True,CPU_only=True,torch_version=torch.__version__,torch_package_version=__import__('importlib.metadata',fromlist=['version']).version('torch'),actual_PyTorch_forward_hooks=True,
  endpoint_passthrough=True,plain_DS_off=True,eadp_DS_on=True,last_observer_saw_effective_kwargs=True,continuation_untouched=True,
  missing_mask_kept_original_call=True,wrong_cardinality_and_dtype_rejected=True,model_loaded=False,CUDA_initialized=False,GT_read=False,scoring_started=False,
  plan_sha256=c.sha(c.PLAN))
target='cpu_torch_hook_test_author.json' if '--author' in sys.argv else 'cpu_torch_hook_test_llava.json'
c.save(c.HERE/target,report);print(json.dumps(report))
