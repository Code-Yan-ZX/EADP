"""CPU-testable local Qwen3 StackTok adaptation primitives; no GPU entry point.

This is not the author's Qwen3 implementation and does not patch a model.
"""
import torch

def group_premerger(hidden,grid_thw,merge_size=2):
 if type(merge_size) is not int or merge_size!=2 or not isinstance(grid_thw,(tuple,list)) or len(grid_thw)!=3:raise ValueError('unsupported merge geometry')
 t,h,w=grid_thw
 if any(type(x) is not int or x<=0 for x in grid_thw) or h%merge_size or w%merge_size or hidden.ndim!=2 or hidden.shape[0]!=t*h*w:raise ValueError('inconsistent visual grid')
 return hidden.reshape(-1,merge_size*merge_size,hidden.shape[-1]).mean(dim=1)

def user_question(message):
 if not isinstance(message,list):raise ValueError('message must be list')
 parts=[]
 for item in message:
  if item.get('type')=='text':
   value=item.get('value')
   if not isinstance(value,str):raise ValueError('invalid text part')
   if value:parts.append(value)
 question=' '.join(parts).strip().replace('<image>','').strip()
 if not question:raise ValueError('missing user question')
 return 'Question: '+question

def author_selection(selector,text,projected,premerge,k,tv_temp=.01,vv_temp=.2):
 if projected.ndim!=2 or premerge.ndim!=2 or projected.shape[0]!=premerge.shape[0] or text.ndim!=2 or text.shape[-1]!=projected.shape[-1] or type(k) is not int or not 1<=k<=projected.shape[0]:raise ValueError('selector input shape or budget')
 if k==projected.shape[0]:
  # Matches author StackTokCore.select_vision_tokens keep-all shortcut.
  ids=list(range(k));return {'spatial_ids':ids,'greedy_ids':None,'tokens':projected,'keep_all_shortcut':True}
 if selector.target_vision_tokens!=k:raise ValueError('author selector budget mismatch')
 ids,tokens,info=selector.stacktok_single(text,projected,premerge,tv_temp=tv_temp,vv_temp=vv_temp,padding_patch_indices=None)
 if not isinstance(ids,list) or len(ids)!=k or ids!=sorted(set(ids)) or any(type(x) is not int or not 0<=x<projected.shape[0] for x in ids):raise ValueError('invalid author selection')
 index=torch.tensor(ids,device=projected.device,dtype=torch.long)
 if not torch.equal(tokens,projected.index_select(0,index)):raise ValueError('author output is not exact gathered projection')
 greedy=info.get('selected_before_sort')
 if not isinstance(greedy,list) or sorted(greedy)!=ids:raise ValueError('greedy/spatial set mismatch')
 return {'spatial_ids':ids,'greedy_ids':greedy,'tokens':tokens,'keep_all_shortcut':False}

def p3_ds_prefill(input_ids,attention_mask,text_embeds,full_positions,rope_delta,visual_start,visual,deepstack,spatial_ids,image_token_id,*,decode_cache_basis):
 # EADP's compact-prompt generation and native full-ID generation do not use
 # the same nominal decode cache position.  The caller must declare the basis;
 # neither branch is a substitute for real fixed-prefix position/KV tests.
 if decode_cache_basis not in ('compact','original'):raise ValueError('decode cache basis must be explicit')
 if input_ids.ndim!=2 or input_ids.shape[0]!=1 or attention_mask.shape!=input_ids.shape or text_embeds.ndim!=3 or text_embeds.shape[:2]!=input_ids.shape or full_positions.shape!=(3,1,input_ids.shape[1]):raise ValueError('batch or position shape')
 n=visual.shape[0];length=input_ids.shape[1];k=len(spatial_ids)
 if type(visual_start) is not int or visual_start<0 or visual_start+n>length or visual.ndim!=2 or visual.shape[1]!=text_embeds.shape[2] or not isinstance(deepstack,(list,tuple)) or len(deepstack)!=3 or any(x.shape!=visual.shape for x in deepstack):raise ValueError('visual stream shape')
 if k<1 or spatial_ids!=sorted(set(spatial_ids)) or any(type(x) is not int or not 0<=x<n for x in spatial_ids):raise ValueError('invalid spatial IDs')
 expected=torch.zeros_like(input_ids,dtype=torch.bool);expected[:,visual_start:visual_start+n]=True
 if not torch.equal(input_ids==image_token_id,expected):raise ValueError('noncontiguous or mismatched image tokens')
 device=input_ids.device;idx=torch.tensor(spatial_ids,device=device,dtype=torch.long);retained=torch.cat((torch.arange(visual_start,device=device),visual_start+idx,torch.arange(visual_start+n,length,device=device)))
 embeds=text_embeds.clone();embeds[:,visual_start:visual_start+n]=visual.to(dtype=embeds.dtype).unsqueeze(0);embeds=embeds.index_select(1,retained);mask=attention_mask.index_select(1,retained);positions=full_positions.index_select(2,retained);visual_mask=torch.zeros_like(mask,dtype=torch.bool);visual_mask[:,visual_start:visual_start+k]=True
 streams=[v.index_select(0,idx.to(v.device)) for v in deepstack]
 adjusted_delta=rope_delta+(n-k) if decode_cache_basis=='compact' else rope_delta
 return {'inputs_embeds':embeds,'attention_mask':mask,'position_ids':positions,'visual_pos_masks':visual_mask,'deepstack_visual_embeds':streams,'rope_delta':adjusted_delta,'retained_indices':retained,'decode_cache_basis':decode_cache_basis}
