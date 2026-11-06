"""Read-only model hooks for PriorTR diagnostic; never alter selections or math."""
import torch

class Trace:
 def __init__(self,model,forced=False):self.forced=forced;self.model=model;self.calls=[];self.layers=[];self.injections=[];self.prunes=[];self.arrays={};self.handles=[];self.undo=[];self.step=-1
 def save(self,name,t):
  if t is not None:self.arrays[name]=t.detach().contiguous().cpu().clone()
 def __enter__(self):
  lm=self.model.model.language_model
  def vision(_m,_args,out):
   self.save('full_visual',out.pooler_output)
   for i,x in enumerate(out.deepstack_features or []):self.save('vision_ds_'+str(i),x)
  self.handles.append(self.model.model.visual.register_forward_hook(vision))
  def pre(_m,args,kw):
   self.step+=1;pos=kw.get('position_ids');emb=kw.get('inputs_embeds');ids=kw.get('input_ids');cache=kw.get('past_key_values')
   lens=[] if cache is None else [None if x.keys is None else int(x.keys.shape[2]) for x in cache.layers]
   self.calls.append({'step':self.step,'tokens':int((emb if emb is not None else ids).shape[1]),'cache_before':lens,'position_ids':None if pos is None else pos.detach().cpu().tolist()})
   if self.step==0:
    self.save('original_prefill_position',pos);self.save('original_prefill_embeds',emb);self.save('original_visual_mask',kw.get('visual_pos_masks'))
    for i,x in enumerate(kw.get('deepstack_visual_embeds') or []):self.save('full_ds_'+str(i),x)
  self.handles.append(lm.register_forward_pre_hook(pre,with_kwargs=True))
  def post(_m,_args,out):
   cache=out.past_key_values
   self.calls[-1]['cache_after']=[] if cache is None else [None if x.keys is None else int(x.keys.shape[2]) for x in cache.layers]
  self.handles.append(lm.register_forward_hook(post))
  for idx,layer in enumerate(lm.layers):
   def block(_m,args,kw,idx=idx):
    hidden=args[0] if args else kw['hidden_states'];self.layers.append({'step':self.step,'layer':idx,'tokens':int(hidden.shape[1]),'mode':'normal'})
   self.handles.append(layer.register_forward_pre_hook(block,with_kwargs=True))
  def logits(_m,_args,out):
   if self.step==0 or self.forced:self.save('logits_'+str(self.step),out)
  self.handles.append(self.model.lm_head.register_forward_hook(logits))
  original_ds=lm._deepstack_process
  def observe_ds(hidden,mask,embed):
   i=len(self.injections);self.injections.append({'step':self.step,'sequence':hidden.shape[1],'rows':embed.shape[0],'mask_count':int(mask.sum())});self.save('injected_ds_'+str(i),embed);self.save('injected_mask_'+str(i),mask)
   return original_ds(hidden,mask,embed)
  lm._deepstack_process=observe_ds;self.undo.append((lm,'_deepstack_process',original_ds))
  if hasattr(lm,'_score_layer_forward'):
   original=lm._score_layer_forward
   def score(layer,hidden_states,*args,**kw):
    self.layers.append({'step':self.step,'layer':layer.self_attn.layer_idx,'tokens':int(hidden_states.shape[1]),'mode':'score_lookahead'})
    return original(layer,hidden_states,*args,**kw)
   lm._score_layer_forward=score;self.undo.append((lm,'_score_layer_forward',original))
   original_prune=lm._apply_pruning
   def prune(*args,**kw):
    j=len(self.prunes);self.save('prune_before_position_'+str(j),kw['position_ids']);self.save('prune_before_hidden_'+str(j),kw['hidden_states'])
    out=original_prune(*args,**kw);self.save('prune_after_hidden_'+str(j),out[0]);self.save('prune_after_position_'+str(j),out[1]);self.save('prune_after_mask_'+str(j),out[7])
    self.prunes.append({'layer':kw['layer_idx'],'before':int(kw['hidden_states'].shape[1]),'after':int(out[0].shape[1]),'range_before':list(kw['image_token_range']),'range_after':list(out[9])});return out
   lm._apply_pruning=prune;self.undo.append((lm,'_apply_pruning',original_prune))
   strategy=self.model.vtr_strategy;original_select=strategy.select_tokens
   def choose(scores,n,config,layer_idx=0):
    ids=original_select(scores,n,config,layer_idx);self.save('selection_scores_'+str(len(self.prunes)),scores);self.save('selected_ids_'+str(len(self.prunes)),ids);return ids
   strategy.select_tokens=choose;self.undo.append((strategy,'select_tokens',original_select))
  return self
 def __exit__(self,*exc):
  for h in self.handles:h.remove()
  for obj,name,original in reversed(self.undo):setattr(obj,name,original)
  return False
