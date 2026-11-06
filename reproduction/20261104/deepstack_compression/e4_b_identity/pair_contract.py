"""Exact same-input C-DROP/C-SKIP versus accepted E4 C-ON/off anchors."""
from pathlib import Path
import sys
E4=Path(__file__).resolve().parent.parent/'e4';sys.path.insert(1,str(E4))
from identity_records import compare_pair
import re

SCALARS=('do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty')

def _fingerprints(record,forced):
    trace=record.get('decode_trace');kv=record.get('cache_fingerprints');count=33 if forced else 1
    if not isinstance(trace,list) or len(trace)!=len(record['output_ids']) or not isinstance(kv,list) or len(kv)!=count:
        raise ValueError('decode/KV evidence incomplete')
    for i,step in enumerate(kv):
        length=len(record['retained_ids'])+i
        if (not isinstance(step,list) or len(step)!=36 or trace[i].get('layer_cache_lengths_after')!=[length]*36):
            raise ValueError('invalid actual cache lengths')
        for layer in step:
            if not isinstance(layer,dict) or set(layer)!={'key','value'}:raise ValueError('invalid KV fields')
            for value in layer.values():
                if (not isinstance(value,dict) or set(value)!={'dtype','shape','sha256'} or value['dtype']!='torch.bfloat16' or
                    value['shape']!=[1,8,length,128] or not isinstance(value['sha256'],str) or
                    re.fullmatch('[0-9a-f]{64}',value['sha256']) is None):raise ValueError('invalid KV fingerprint')

def _endpoint(record):
    ids=record.get('output_ids');eos=record.get('eos_ids');limit=record.get('max_new_tokens')
    if (not isinstance(ids,list) or not ids or any(type(x) is not int for x in ids) or
        not isinstance(eos,list) or any(type(x) is not int for x in eos) or limit!=2048 or len(ids)>limit or
        record.get('eos')!=bool(ids[-1] in eos) or
        record.get('truncated')!=bool(len(ids)==limit and ids[-1] not in eos) or
        not (record['eos'] or record['truncated'])):
        raise ValueError('invalid natural generation endpoint')

def validate_pair(drop,skip,on_ref,off_ref):
    _endpoint(drop);_endpoint(skip);result=compare_pair(drop,skip)
    if drop.get('min_new_tokens')!=skip.get('min_new_tokens'):raise ValueError('actual natural min token setting differs')
    for r in (drop,skip):
        _fingerprints(r,False)
    if (on_ref.get('budget')!=256 or on_ref.get('ds_on') is not True or on_ref.get('forced') is not False or
        off_ref.get('budget')!=256 or off_ref.get('ds_on') is not False or off_ref.get('forced') is not False):
        raise ValueError('expected accepted K256 ON/OFF natural references')
    same=('task','qid','group_id','image_sha256','message_sha256','processor','main_visual','pruned_visual',
          'ordered_ids','spatial_ids','retained_ids','full_position_ids','full_delta','prefill_position_ids',
          'adjusted_delta','prefill_embeds','prefill_mask')
    for ref in (on_ref,off_ref):
        if (set(drop.get('generation_scalars',{}))!=set(SCALARS) or
            drop['generation_scalars']!=ref.get('generation_scalars') or
            skip.get('generation_scalars')!=ref.get('generation_scalars') or
            drop['generation_scalars'].get('do_sample') is not False or drop['generation_scalars'].get('max_new_tokens')!=2048):
            raise ValueError('actual natural generation recipe differs')
        for key in same:
            if ref.get(key)!=drop.get(key):raise ValueError('reference mismatch: '+key)
    for key in ('output_ids','answer','eos_ids','eos','truncated','max_new_tokens'):
        if off_ref.get(key)!=drop.get(key):raise ValueError('OFF natural endpoint mismatch: '+key)
    if off_ref.get('prefill_logits') is not None and off_ref['prefill_logits']!=drop.get('first_logits'):
        raise ValueError('OFF prefill logits mismatch')
    if drop.get('decode_trace')!=skip.get('decode_trace') or drop.get('cache_fingerprints')!=skip.get('cache_fingerprints'):
        raise ValueError('natural decode/KV fingerprint mismatch')
    return result

def validate_forced_pair(drop,skip,natural_drop,natural_skip,token):
    if type(token) is not int or token<0:raise ValueError('invalid fixed token')
    for record,natural in ((drop,natural_drop),(skip,natural_skip)):
        _fingerprints(record,True)
        if (record.get('forced') is not True or record.get('forced_token')!=token or
            record.get('output_ids')!=[token]*33 or record.get('eos_ids')!=[] or record.get('eos') is not False or
            record.get('truncated') is not False or record.get('max_new_tokens')!=33 or
            len(record.get('decode_trace',[]))!=33 or len(record.get('cache_fingerprints',[]))!=33 or
            any(not isinstance(step,list) or len(step)!=36 for step in record.get('cache_fingerprints',[]))):
            raise ValueError('forced 32-step evidence incomplete')
        expected=dict(natural['generation_scalars']);expected['max_new_tokens']=33
        if record.get('generation_scalars')!=expected or record.get('min_new_tokens')!=33:
            raise ValueError('forced generation recipe differs')
        for key in ('task','qid','group_id','image_sha256','message_sha256','processor','main_visual','pruned_visual',
                    'ordered_ids','spatial_ids','retained_ids','full_position_ids','full_delta','prefill_position_ids',
                    'adjusted_delta','prefill_embeds','prefill_mask','first_logits'):
            if record.get(key)!=natural.get(key):raise ValueError('forced/natural prefill mismatch: '+key)
    result=compare_pair(drop,skip)
    if drop['decode_trace']!=skip['decode_trace'] or drop['cache_fingerprints']!=skip['cache_fingerprints']:
        raise ValueError('actual forced decode/KV fingerprint mismatch')
    return result
