"""Local Qwen3 StackTok hook prototype, diagnostics only; no standalone GPU entry."""
from contextlib import ExitStack
import torch
from local_adapter import author_selection,p3_ds_prefill
from runtime_capture import VisualCapture,encode_question

class FullStackTokAdapter:
 """Summary-only Qwen3 adapter; keep-alive and identity depend on three-anchor proof."""
 def __init__(self,wrapper,row,selector,k,reference):
  if k not in (256,1024):raise ValueError('unsupported diagnostic K')
  self.w=wrapper;self.row=row;self.selector=selector;self.k=k;self.reference=reference;self.stack=ExitStack();self.state={'calls':0};self.entered=False
 def __enter__(self):
  if self.entered:raise ValueError('adapter context cannot be reused')
  self.entered=True;w=self.w;previous=w._processor_inputs;had='_processor_inputs' in w.__dict__;raw=w.__dict__.get('_processor_inputs')
  def restore():
   if had:w._processor_inputs=raw
   else:delattr(w,'_processor_inputs')
  def processor(*args,**kw):
   if 'inputs' in self.state:raise ValueError('multiple processor calls')
   x=previous(*args,**kw);required=('input_ids','attention_mask','pixel_values','image_grid_thw')
   self.state['inputs']={name:x[name].detach().clone() for name in required}
   grid=self.state['inputs']['image_grid_thw'];
   if grid.shape!=(1,3):raise ValueError('single image grid required')
   self.state['capture']=self.stack.enter_context(VisualCapture(w.model.model.visual,tuple(int(v) for v in grid[0].tolist())))
   return x
  try:
   w._processor_inputs=processor;self.stack.callback(restore)
   h=w.model.model.language_model.register_forward_pre_hook(self._before,with_kwargs=True);self.stack.callback(h.remove)
  except BaseException:self.stack.close();raise
  return self
 def _before(self,_module,args,kw):
  self.state['calls']+=1;inp=self.state.get('inputs');emb=kw.get('inputs_embeds');cache=kw.get('cache_position');mask=kw.get('attention_mask');pos=kw.get('position_ids')
  if inp is None or emb is None or cache is None or mask is None or pos is None:raise ValueError('missing actual inputs or generation coordinates')
  ids=inp['input_ids'];length=ids.shape[1]
  if self.state['calls']==1:
   cap=self.state['capture'].result();main=cap['postmerger'];ds=cap['deepstack'];n=main.shape[0];token=self.w.model.config.image_token_id;where=torch.nonzero(ids[0]==token,as_tuple=True)[0]
   if n!=1024 or where.numel()!=n or not torch.equal(where,torch.arange(where[0],where[0]+n,device=where.device)):raise ValueError('expected contiguous 1024 image tokens')
   start=int(where[0].item());actual_ds=kw.get('deepstack_visual_embeds');vm=kw.get('visual_pos_masks')
   if not isinstance(actual_ds,(tuple,list)) or len(actual_ds)!=3 or any(not torch.equal(x,y) for x,y in zip(actual_ds,ds)) or vm is None or not torch.equal(vm,ids==token) or not torch.equal(emb[0,start:start+n],main.to(emb.dtype)):raise ValueError('native prefill main/DS mismatch')
   from full_refs import validate_before_selection
   validate_before_selection(inp,main,ds,pos,self.w.model.model.rope_deltas,start,self.reference)
   question=encode_question(self.w.processor.tokenizer,self.w.model.get_input_embeddings(),self.row['message'],ids.device)
   selected=author_selection(self.selector,question['embeddings'],main,cap['averaged_premerger'],self.k)
   if not torch.equal(cache,torch.arange(length,device=cache.device,dtype=cache.dtype)) or not torch.equal(mask,inp['attention_mask']):raise ValueError('unexpected prefill coordinates')
   delta=self.w.model.model.rope_deltas.detach().clone();out=p3_ds_prefill(ids,mask,emb,pos,delta,start,main,ds,selected['spatial_ids'],token,decode_cache_basis='original')
   # The language model may update its input tensor in place; diagnostic
   # snapshots must reflect the values at the actual pre-forward boundary.
   snap={name:([x.detach().clone() for x in value] if isinstance(value,list) else value.detach().clone() if torch.is_tensor(value) else value) for name,value in out.items()}
   self.state.update(question=question,selected=selected,original_delta=delta,original_length=length,removed=n-self.k,retained=out['retained_indices'],prefill=snap,visual=cap,decode_count=0)
   if self.k==n:return None
   kw=dict(kw);kw.update({k:out[k] for k in ('inputs_embeds','attention_mask','position_ids','visual_pos_masks','deepstack_visual_embeds')});kw['cache_position']=torch.arange(length-self.state['removed'],device=cache.device,dtype=cache.dtype)
   return args,kw
  if self.k==1024:return None
  removed=self.state.get('removed');retained=self.state.get('retained');step=self.state['decode_count']
  nominal=length+step;physical=nominal-removed
  if removed is None or emb.shape[1]!=1 or cache.numel()!=1 or int(cache.item())!=nominal or mask.shape!=(1,nominal+1):raise ValueError('unexpected decode cache sequence or mask shape')
  if not torch.equal(mask[:,:length],inp['attention_mask']) or not torch.all(mask[:,length:]==1):raise ValueError('decode mask changed prefix or generated columns')
  expected_pos=torch.full((3,1,1),nominal+int(self.state['original_delta'].item()),device=pos.device,dtype=pos.dtype)
  if pos.shape!=(3,1,1) or not torch.equal(pos,expected_pos):raise ValueError('unexpected decode position')
  past=kw.get('past_key_values')
  if past is None or any(past.get_seq_length(i)!=physical for i in range(36)):raise ValueError('unexpected decode KV lengths')
  extra=torch.arange(length,mask.shape[1],device=mask.device);indices=torch.cat((retained.to(mask.device),extra))
  kw=dict(kw);kw['attention_mask']=mask.index_select(1,indices);kw['cache_position']=cache-removed
  self.state['decode_count']+=1
  return args,kw
 def __exit__(self,*args):self.stack.close()
