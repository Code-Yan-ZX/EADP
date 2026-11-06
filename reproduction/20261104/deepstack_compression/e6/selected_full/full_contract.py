"""Independent all-record summaries; full tensor bytes are not reconstructible."""
import hashlib,re
from full_imports import prepare,load,HERE
def _meta(v):
 return (isinstance(v,dict) and set(v)=={'dtype','shape','sha256'} and isinstance(v['dtype'],str) and
         isinstance(v['shape'],list) and v['shape'] and all(type(n) is int and n>0 for n in v['shape']) and
         isinstance(v['sha256'],str) and re.fullmatch('[0-9a-f]{64}',v['sha256']) is not None)
def _endpoint(r,tokenizer,reference):
 ids=r.get('output_ids');eos=r.get('eos_ids')
 if (r.get('forced') is not False or not isinstance(ids,list) or not ids or len(ids)>2048 or any(type(x) is not int for x in ids) or
     not isinstance(eos,list) or any(type(x) is not int for x in eos) or eos!=reference['eos_ids'] or
     r.get('eos')!=bool(ids[-1] in eos) or r.get('truncated')!=bool(len(ids)==2048 and ids[-1] not in eos) or
     not(r['eos'] or r['truncated']) or r.get('generation_scalars')!=reference['generation_scalars'] or
     tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)!=r.get('answer')):
  raise ValueError('natural output or recipe mismatch')
def validate_stock(r,reference,tokenizer):
 import numpy as np
 prepare();from decode_observer import validate_trace
 if r.get('path')!='author_stock' or r.get('method') not in ('div','cd') or r.get('injected_mask') is not None or r.get('injected_deepstack') is not None:
  raise ValueError('not unmodified author stock')
 start=r['image_start'];length=r['original_length'];ids=r['spatial_ids'];retained=r['retained_ids']
 if (type(start) is not int or type(length) is not int or start<0 or start+1024>length or len(ids)!=256 or
     any(type(i) is not int or not 0<=i<1024 for i in ids) or ids!=sorted(set(ids)) or
     retained!=list(range(start))+[start+i for i in ids]+list(range(start+1024,length))):raise ValueError('stock selected indices')
 for left,right in (('processor','processor'),('main_visual','visual_sha256'),('full_deepstack','deepstack_full_sha256'),
                    ('full_position_ids','original_position_ids'),('full_delta','full_rope_delta'),('image_start','image_start'),('original_length','original_length')):
  if r[left]!=reference[right]:raise ValueError('stock E1 input differs: '+left)
 metas={'visual':r['main_visual'],'pruned':r['pruned_visual'],'prefill':r['prefill_embeds'],'prefill_logits':r['prefill_logits'],
        **{'ds'+str(i):v for i,v in enumerate(r['full_deepstack'])},**{'processor_'+k:v for k,v in r['processor'].items()}}
 if len(r['full_deepstack'])!=3 or any(not _meta(x) for x in metas.values()) or r['arrays']!={'kind':'summary_only','tensors':metas}:
  raise ValueError('stock tensor summaries incomplete')
 visual=r['main_visual'];pruned=r['pruned_visual'];prefill=r['prefill_embeds']
 if (visual['dtype']!='torch.bfloat16' or len(visual['shape'])!=2 or visual['shape'][0]!=1024 or
     pruned['dtype']!=visual['dtype'] or pruned['shape']!=[256,visual['shape'][1]] or
     prefill['dtype']!=visual['dtype'] or prefill['shape']!=[1,len(retained),visual['shape'][1]]):raise ValueError('stock feature shapes')
 mask=np.ones((1,len(retained)),dtype=np.int64)
 if r['prefill_mask']!={'dtype':'torch.int64','shape':[1,len(retained)],'sha256':hashlib.sha256(mask.tobytes()).hexdigest()}:
  raise ValueError('stock prefill mask')
 _endpoint(r,tokenizer,reference);validate_trace(r['decode_trace'],r['output_ids'],len(retained),r['rope_delta'][0][0])
 if r['rope_delta']!=[[0]] or r['prefill_position_ids']!=[[[i for i in range(len(retained))]]]*3:
  raise ValueError('stock P1 positions or delta differ')
 if r['decode_trace'][0]['position_ids']!=r['prefill_position_ids']:raise ValueError('stock actual position trace')
 return True
def validate_adapted(r,reference,tokenizer):
 _stock,contract=prepare();from verify_k256 import validate_budget_summary
 if (r.get('method') not in ('div','cd') or r.get('selection_mode')!='author_'+r['method'] or
     r.get('selector_calls')!=1 or r.get('greedy_calls')!=0 or r.get('ordered_ids')!=r.get('spatial_ids')):
  raise ValueError('not actual author selected baseline')
 _endpoint(r,tokenizer,reference)
 normalized=dict(r,greedy_calls=1,selection_mode='author_greedy')
 validate_budget_summary(normalized,reference,tokenizer)
 return True
