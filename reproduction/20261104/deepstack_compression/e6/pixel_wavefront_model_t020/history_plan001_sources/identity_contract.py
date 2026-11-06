"""Fail-closed identity contract for the future t=.2 model diagnostic.

CPU validation only: these functions do not establish that a model executed.
A separate independently-reviewed acceptance process must reconstruct the
archived actual tensors and bind every record to real supervised GPU execution.
"""
import re

SHARED=('task','qid','group_id','image_sha256','message_sha256','processor','generation_scalars','eos_ids','original_sequence_length','original_max_position',
        'output_ids','selected_indices','prefill_mask','prefill_position_ids','rope_delta','visual_mask',
        'main_visual','injected_deepstack','prefill_embeddings','prefill_logits',
        'decode_trace','cache_fingerprints')

def _metadata(x):
 return (isinstance(x,dict) and isinstance(x.get('dtype'),str) and x['dtype'] and
         isinstance(x.get('shape'),list) and all(type(q) is int and q>=0 for q in x['shape']) and
         isinstance(x.get('sha256'),str) and re.fullmatch('[0-9a-f]{64}',x['sha256']) is not None)

def validate_record(x,anchor,recipe):
 if not isinstance(x,dict) or any(k not in x for k in SHARED):raise ValueError('missing required identity evidence')
 if x.get('status')!='completed' or x.get('path') not in ('author','wavefront') or any(x.get(k)!=anchor.get(k) for k in ('task','qid','group_id','image_sha256','message_sha256')):raise ValueError('arm/input mismatch')
 if x['generation_scalars']!=recipe or x.get('GT_opened') is not False or x.get('scoring_performed') is not False:raise ValueError('recipe/scope mismatch')
 indices=x['selected_indices'];ids=x['output_ids'];eos=x['eos_ids']
 if not isinstance(indices,list) or not indices or indices!=sorted(set(indices)) or any(type(i) is not int or not 0<=i<1024 for i in indices):raise ValueError('selection invalid')
 if not isinstance(ids,list) or not ids or len(ids)>recipe['max_new_tokens'] or any(type(i) is not int or i<0 for i in ids) or not isinstance(eos,list) or not eos or not (ids[-1] in eos or len(ids)==recipe['max_new_tokens']):raise ValueError('natural generation incomplete')
 for field in ('main_visual','prefill_embeddings','prefill_logits'):
  if not _metadata(x[field]):raise ValueError('missing tensor metadata: '+field)
 for field in ('injected_deepstack',):
  if not isinstance(x[field],list) or len(x[field])!=3 or not all(_metadata(z) for z in x[field]):raise ValueError('three DS tensors required')
 if not isinstance(x['processor'],dict) or any(not _metadata(x['processor'].get(k)) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')):raise ValueError('processor identity incomplete')
 if not isinstance(x['prefill_position_ids'],list) or len(x['prefill_position_ids'])!=3 or not isinstance(x['prefill_mask'],list) or not isinstance(x['visual_mask'],list) or not isinstance(x['rope_delta'],list):raise ValueError('position or mask missing')
 trace=x['decode_trace'];fp=x['cache_fingerprints']
 if not isinstance(trace,list) or len(trace)!=len(ids) or not isinstance(fp,list) or len(fp)!=len(ids):raise ValueError('per-step trace missing')
 seq=x['original_sequence_length'];maxpos=x['original_max_position'];n=len(indices)
 if type(seq) is not int or seq<1024 or type(maxpos) is not int or maxpos<0 or x['processor']['input_ids']['shape']!=[1,seq]:raise ValueError('invalid original input shape')
 initial=seq-1024+n
 if x['prefill_mask']!=[[1]*initial] or len(x['visual_mask'])!=1 or len(x['visual_mask'][0])!=initial or sum(x['visual_mask'][0])!=n:raise ValueError('invalid prefill masks')
 if any(len(axis)!=1 or len(axis[0])!=initial for axis in x['prefill_position_ids']):raise ValueError('invalid prefill position shape')
 for i,step in enumerate(fp):
  if not isinstance(step,list) or len(step)!=36 or any(not isinstance(kv,dict) or set(kv)!={'key','value'} or not _metadata(kv['key']) or not _metadata(kv['value']) for kv in step):raise ValueError('36-layer KV fingerprint missing')
  if any(len(kv[name]['shape'])<3 or kv[name]['shape'][-2]!=initial+i for kv in step for name in ('key','value')):raise ValueError('cache length trajectory')
  t=trace[i]
  if not isinstance(t,dict) or t.get('rope_delta')!=x['rope_delta']:raise ValueError('missing step/rope')
  if i==0:
   if t.get('position_ids')!=x['prefill_position_ids'] or t.get('cache_position')!=list(range(initial)):raise ValueError('prefill position trajectory')
  elif t.get('position_ids')!=[[[maxpos+i]]]*3 or t.get('cache_position')!=[seq+i-1]:raise ValueError('decode position trajectory')
 if not isinstance(x.get('archive_path'),str) or not re.fullmatch('[0-9a-f]{64}',x.get('archive_sha256','')):raise ValueError('missing actual tensor archive')
 return True

def compare_pair(author,wavefront,anchor,recipe):
 validate_record(author,anchor,recipe);validate_record(wavefront,anchor,recipe)
 if author['path']!='author' or wavefront['path']!='wavefront':raise ValueError('arm order')
 different=[k for k in SHARED if author[k]!=wavefront[k]]
 if different:raise ValueError('mismatched same-input model evidence: '+','.join(different))
 return True

def validate_pair_set(pairs,anchors,recipe):
 if not isinstance(pairs,list) or len(pairs)!=3 or len(anchors)!=3:raise ValueError('three complete pairs required')
 if len({(a.get('task'),a.get('qid'),a.get('group_id')) for a in anchors})!=3 or len({a.get('group_id') for a in anchors})!=3:raise ValueError('three distinct image identities required')
 for pair,anchor in zip(pairs,anchors):
  if not isinstance(pair,(tuple,list)) or len(pair)!=2:raise ValueError('pair incomplete')
  compare_pair(pair[0],pair[1],anchor,recipe)
 return True

def admit_timing(*_args,**_kwargs):
 raise RuntimeError('timing locked: independent source-bound approval and actual completed model identity are not implemented')
