"""No-GT, read-only stock/selected identity and full-array contracts."""
import hashlib,math,re,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];E4=ROOT/'e4';sys.path.insert(0,str(E4))
from decode_observer import validate_trace
from verify_k256 import validate_budget_record

def compare_pair(stock,adapted):
    if (stock.get('path')!='author_stock' or adapted.get('selection_mode')!='author_'+str(stock.get('method')) or
        adapted.get('method')!=stock.get('method') or type(stock.get('forced')) is not bool or adapted.get('forced') is not stock.get('forced')):
        raise ValueError('incorrect stock/adapted paths')
    if stock.get('forced_token')!=adapted.get('forced_token'):raise ValueError('forced token differs')
    common=('task','qid','group_id','image_sha256','message_sha256','processor','main_visual','full_deepstack',
            'pruned_visual','spatial_ids','retained_ids','full_position_ids','full_delta','prefill_embeds','prefill_mask','generation_scalars','generation_controls')
    for k in common:
        if stock.get(k)!=adapted.get(k):raise ValueError('selector/input identity differs: '+k)
    if stock.get('injected_mask') is not None or stock.get('injected_deepstack') is not None:
        raise ValueError('stock author path unexpectedly injects DS')
    if (adapted.get('selector_calls')!=1 or adapted.get('greedy_calls')!=0 or adapted.get('ds_on') is not True or
        adapted.get('actual_injection_verified') is not True):raise ValueError('adapted selector/DS evidence missing')
    # Output and first logits can differ: stock uses original P1/DS-off and
    # adapted uses P3/DS-on. Their equality is never an admission requirement.
    return True

def validate_adapted(record,reference,arrays,tokenizer):
    method=record.get('method')
    if method not in ('div','cd') or record.get('selection_mode')!='author_'+method or record.get('selector_calls')!=1 or record.get('greedy_calls')!=0:
        raise ValueError('not observed Div/CD author selection')
    if record.get('ordered_ids')!=record.get('spatial_ids'):raise ValueError('author output order is not spatial')
    ids=record.get('output_ids');eos=record.get('eos_ids');forced=record.get('forced')
    if (type(forced) is not bool or not isinstance(ids,list) or not ids or any(type(x) is not int for x in ids) or
        not isinstance(eos,list) or any(type(x) is not int for x in eos) or len(ids)>(33 if forced else 2048) or
        record.get('eos')!=bool(ids[-1] in eos) or record.get('truncated')!=bool(not forced and len(ids)==2048 and ids[-1] not in eos) or
        (forced and (record.get('forced_token') is None or ids!=[record['forced_token']]*33 or eos!=[])) or
        (not forced and ids[-1] not in eos and len(ids)!=2048)):
        raise ValueError('adapted generation endpoint invalid')
    # Reuse E4 tensor reconstruction after explicitly asserting the actual
    # baseline selector. Its generic header labels non-K1024 as EADP greedy;
    # those labels are normalized only in this in-memory validation copy.
    normalized=dict(record,greedy_calls=1,selection_mode='author_greedy')
    return validate_budget_record(normalized,reference,arrays,tokenizer)

def validate_stock(record,arrays,tokenizer):
    if record.get('path')!='author_stock' or record.get('method') not in ('div','cd') or type(record.get('forced')) is not bool:
        raise ValueError('invalid stock record')
    expected={'visual','pruned','prefill','prefill_logits','ds0','ds1','ds2',
              'processor_input_ids','processor_attention_mask','processor_pixel_values','processor_image_grid_thw'}
    if set(arrays)!=expected:raise ValueError('stock array inventory')
    def check(name,meta):
        arr=arrays[name]
        sizes={'torch.bfloat16':2,'torch.float16':2,'torch.float32':4,'torch.float64':8,'torch.int64':8,'torch.int32':4,'torch.uint8':1,'torch.bool':1}
        shape=meta.get('shape');size=sizes.get(meta.get('dtype'))
        if (size is None or not isinstance(shape,list) or not shape or any(type(n) is not int or n<=0 for n in shape)
            or arr.dtype!=np.uint8 or list(arr.shape)!=shape[:-1]+[shape[-1]*size]
            or hashlib.sha256(arr.tobytes()).hexdigest()!=meta.get('sha256')):raise ValueError('stock tensor shape or hash '+name)
        return arr
    full=check('visual',record['main_visual']);pruned=check('pruned',record['pruned_visual']);pf=check('prefill',record['prefill_embeds'])
    for i,meta in enumerate(record['full_deepstack']):check('ds'+str(i),meta)
    for key,meta in record['processor'].items():check('processor_'+key,meta)
    logits=check('prefill_logits',record['prefill_logits']);dtype=record['prefill_logits']['dtype'];raw=logits.tobytes()
    if dtype=='torch.bfloat16':vals=(np.frombuffer(raw,dtype='<u2').astype(np.uint32)<<16).view(np.float32)
    elif dtype in ('torch.float16','torch.float32','torch.float64'):vals=np.frombuffer(raw,dtype={'torch.float16':'<f2','torch.float32':'<f4','torch.float64':'<f8'}[dtype])
    else:raise ValueError('stock logits dtype')
    if not vals.size or not np.isfinite(vals).all():raise ValueError('stock nonfinite logits')
    ids=record['spatial_ids'];start=record['image_start'];length=record['original_length']
    if (type(start) is not int or type(length) is not int or start<0 or start+1024>length or
        full.shape[0]!=1024 or len(ids)!=256 or any(type(i) is not int for i in ids) or
        ids!=sorted(set(ids)) or ids[0]<0 or ids[-1]>=1024 or not np.array_equal(pruned,full[ids])):raise ValueError('stock pruned rows')
    retained=list(range(start))+[start+i for i in ids]+list(range(start+1024,length))
    if record['retained_ids']!=retained:raise ValueError('stock retained IDs')
    buckets={}
    for i,line in enumerate(full):buckets.setdefault(hashlib.sha256(line.tobytes()).digest(),[]).append(i)
    if any(len([i for i in buckets[hashlib.sha256(line.tobytes()).digest()] if np.array_equal(full[i],line)])!=1 for line in pruned):
        raise ValueError('stock selected feature ambiguous')
    if not np.array_equal(pf[0,start:start+256],pruned):raise ValueError('stock actual prefill visual differs')
    ids_out=record['output_ids'];eos=bool(ids_out and ids_out[-1] in record['eos_ids']);truncated=not eos and len(ids_out)>=2048
    forced=record['forced']
    if (not ids_out or len(ids_out)>2048 or record['eos']!=eos or record['truncated']!=truncated or
        (forced and (type(record.get('forced_token')) is not int or ids_out!=[record['forced_token']]*33 or record['eos_ids']!=[])) or
        (not forced and not(eos or truncated)) or
        tokenizer.decode(ids_out,skip_special_tokens=True,clean_up_tokenization_spaces=False)!=record['answer']):raise ValueError('stock generation endpoint/decode')
    validate_trace(record['decode_trace'],ids_out,len(record['retained_ids']),record['rope_delta'][0][0])
    if record['decode_trace'][0]['position_ids']!=record['prefill_position_ids']:raise ValueError('stock prefill position trace mismatch')
    return True

def compare_forced(natural,forced):
    if natural.get('forced') is not False or forced.get('forced') is not True:raise ValueError('wrong forced comparison')
    keys=('method','task','qid','group_id','processor','main_visual','full_deepstack','pruned_visual','ordered_ids','spatial_ids',
          'retained_ids','full_position_ids','full_delta','prefill_embeds','prefill_mask','prefill_position_ids','adjusted_delta',
          'injected_mask','injected_deepstack','selected_deepstack','prefill_logits')
    for k in keys:
        if natural.get(k)!=forced.get(k):raise ValueError('forced anchor prefill differs: '+k)
    if 'cache_fingerprints' in natural or 'cache_fingerprints' in forced:
        if natural['cache_fingerprints'][0]!=forced['cache_fingerprints'][0]:raise ValueError('adapted initial KV differs')
    return True

def validate_cache(record):
    forced=record['forced'];trace=record['decode_trace'];kv=record['cache_fingerprints'];count=33 if forced else 1
    if len(kv)!=count or len(trace)!=len(record['output_ids']):raise ValueError('missing actual cache fingerprints')
    for i,step in enumerate(kv):
        n=len(record['retained_ids'])+i
        if len(step)!=36 or trace[i]['layer_cache_lengths_after']!=[n]*36:raise ValueError('bad KV lengths')
        for layer in step:
            if set(layer)!={'key','value'}:raise ValueError('bad KV fields')
            for value in layer.values():
                if (set(value)!={'dtype','shape','sha256'} or value['dtype']!='torch.bfloat16' or
                    value['shape']!=[1,8,n,128] or not isinstance(value['sha256'],str) or
                    re.fullmatch('[0-9a-f]{64}',value['sha256']) is None):raise ValueError('bad KV fingerprint')
    return True

def compare_stock_forced(natural,forced):
    if natural.get('forced') is not False or forced.get('forced') is not True:raise ValueError('wrong forced stock anchor')
    for key in ('method','task','qid','group_id','processor','main_visual','full_deepstack','pruned_visual','spatial_ids',
                'retained_ids','full_position_ids','full_delta','prefill_embeds','prefill_mask','prefill_position_ids',
                'rope_delta','prefill_logits'):
        if natural.get(key)!=forced.get(key):raise ValueError('stock forced prefill differs: '+key)
    if natural['cache_fingerprints'][0]!=forced['cache_fingerprints'][0]:raise ValueError('stock initial KV differs')
    return True
