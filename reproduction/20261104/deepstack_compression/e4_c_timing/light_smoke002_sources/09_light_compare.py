"""CPU-only strict diagnostic comparisons against independently accepted B/E4."""
import hashlib
from pathlib import Path
import numpy as np

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()

COMMON=('task','qid','group_id','image_sha256','message_sha256','processor','main_visual','pruned_visual','image_start','original_length',
        'full_deepstack','ordered_ids','spatial_ids','retained_ids','full_position_ids','full_delta','prefill_position_ids','adjusted_delta',
        'prefill_embeds','prefill_mask','output_ids','answer','eos_ids','eos','truncated','max_new_tokens','generation_scalars',
        'forced','forced_token','decode_trace','block_calls','main_merger_calls','ds_calls')

def compare_metadata(actual,reference,allow_missing_reference_logits=False,allow_missing_reference_kv=False):
 mode=actual.get('mode')
 if mode not in ('C-ON','C-DROP','C-SKIP'):raise ValueError('invalid mode')
 if mode=='C-ON':
  if reference.get('budget')!=256 or reference.get('ds_on') is not True or reference.get('actual_injection_verified') is not True:raise ValueError('wrong E4 ON reference')
 else:
  if reference.get('mode')!=mode or reference.get('actual_ds_off') is not True:raise ValueError('wrong B OFF reference')
 for key in COMMON:
  if key not in actual or key not in reference or actual[key]!=reference[key]:raise ValueError('identity mismatch: '+key)
 expected_logits=reference.get('prefill_logits') if mode=='C-ON' else reference.get('first_logits')
 if actual.get('prefill_logits') is None:raise ValueError('actual prefill logits missing')
 if expected_logits is None and (not allow_missing_reference_logits or mode!='C-ON' or actual.get('forced') is not False):raise ValueError('required reference logits missing')
 if expected_logits is not None and actual['prefill_logits']!=expected_logits:raise ValueError('prefill logits mismatch')
 if actual.get('ds_none')!=[mode=='C-SKIP']*3:raise ValueError('DS execution mismatch')
 if not actual.get('cache_fingerprints'):raise ValueError('actual KV fingerprints missing')
 expected_kv=reference.get('cache_fingerprints')
 if expected_kv is None and (not allow_missing_reference_kv or mode!='C-ON'):raise ValueError('reference KV fingerprints missing')
 if expected_kv is not None and actual['cache_fingerprints']!=expected_kv:raise ValueError('actual KV cache mismatch')
 if mode=='C-ON':
  for key in ('injected_mask','injected_deepstack'):
   if actual.get(key) is None or actual[key]!=reference.get(key):raise ValueError('injected DS mismatch: '+key)
 else:
  if actual.get('injected_mask') is not None or actual.get('injected_deepstack') is not None:raise ValueError('DS off mismatch')
 return {'status':'identity_passed_bitwise','mode':mode,'forced':actual['forced'],'task':actual['task'],'qid':actual['qid'],
         'prefill_logits_compared':expected_logits is not None,'cache_fingerprints_compared':expected_kv is not None}

def compare_archives(actual,reference,allow_missing_reference_logits=False):
 """Compare every reference tensor available in both diagnostic archives."""
 aa=actual['arrays'];rr=reference['arrays'];ap=Path(aa['path']);rp=Path(rr['path'])
 if aa.get('kind')!='full_arrays' or rr.get('kind')!='full_arrays' or sha(ap)!=aa.get('sha256') or sha(rp)!=rr.get('sha256'):raise ValueError('archive identity')
 with np.load(ap,allow_pickle=False) as left,np.load(rp,allow_pickle=False) as right:
  common=set(left.files)&set(right.files);required={'visual','prefill'}
  if not allow_missing_reference_logits:required.add('prefill_logits')
  if not required<=common or ('prefill_logits' in right.files and 'prefill_logits' not in common):raise ValueError('reference arrays missing')
  for name in common:
   if not np.array_equal(left[name],right[name]):raise ValueError('array differs: '+name)
 return {'compared_arrays':sorted(common),'n_compared':len(common)}
