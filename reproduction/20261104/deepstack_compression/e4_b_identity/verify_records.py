"""CPU identity-record checks. Caller must first establish complete run and exit."""
import hashlib
from pathlib import Path
import sys
import numpy as np
from b_imports import local
contract=local('pair_contract');validate_pair=contract.validate_pair;validate_forced_pair=contract.validate_forced_pair
sys.path.insert(1,str(Path(__file__).resolve().parent.parent/'e4'))
from decode_observer import validate_trace

SIZES={'torch.bfloat16':2,'torch.float16':2,'torch.float32':4,'torch.float64':8,
       'torch.int64':8,'torch.int32':4,'torch.uint8':1,'torch.bool':1}

def verify_arrays(record,run):
    item=record['arrays'];path=Path(item['path']).resolve();root=(Path(run)/'arrays').resolve()
    if item.get('kind')!='full_arrays' or path.parent!=root or hashlib.sha256(path.read_bytes()).hexdigest()!=item.get('sha256'):
        raise ValueError('array file identity mismatch')
    meta={'visual':record['main_visual'],'pruned':record['pruned_visual'],'prefill':record['prefill_embeds'],
          'prefill_logits':record['first_logits'],**{'processor_'+k:v for k,v in record['processor'].items()}}
    if record['mode']=='C-DROP':
        if len(record.get('full_deepstack') or [])!=3:raise ValueError('missing actual DS arrays')
        meta.update({'ds'+str(i):x for i,x in enumerate(record['full_deepstack'])})
    elif record['mode']!='C-SKIP' or record.get('full_deepstack') is not None:
        raise ValueError('invalid skip DS state')
    with np.load(path,allow_pickle=False) as arrays:
        if set(arrays.files)!=set(meta):raise ValueError('array inventory mismatch')
        for key,m in meta.items():
            a=arrays[key];size=SIZES.get(m.get('dtype'));n=1
            for d in m.get('shape',[]):n*=d
            if size is None or a.dtype!=np.uint8 or a.size!=n*size or hashlib.sha256(a.tobytes()).hexdigest()!=m.get('sha256'):
                raise ValueError('array contents mismatch: '+key)
        ordered=record['ordered_ids'];spatial=record['spatial_ids'];start=record['image_start'];length=record['original_length']
        if (len(ordered)!=256 or len(set(ordered))!=256 or any(type(x) is not int or x<0 or x>=1024 for x in ordered) or
            spatial!=sorted(ordered) or type(start) is not int or start<0 or length<start+1024 or
            record['retained_ids']!=list(range(start))+[start+x for x in spatial]+list(range(start+1024,length))):
            raise ValueError('invalid actual selector/retained indices')
        vm=meta['visual'];pm=meta['pruned'];em=meta['prefill'];size=SIZES[vm['dtype']]
        if (len(vm['shape'])!=2 or vm['shape'][0]!=1024 or pm['dtype']!=vm['dtype'] or
            pm['shape']!=[256,vm['shape'][1]] or em['dtype']!=vm['dtype'] or
            em['shape']!=[1,len(record['retained_ids']),vm['shape'][1]]):raise ValueError('visual/prefill dimensions mismatch')
        visual=arrays['visual'].reshape(1024,vm['shape'][1],size);pruned=arrays['pruned'].reshape(256,vm['shape'][1],size)
        prefill=arrays['prefill'].reshape(1,len(record['retained_ids']),vm['shape'][1],size)
        if not np.array_equal(visual[spatial],pruned) or not np.array_equal(prefill[0,start:start+256],pruned):
            raise ValueError('actual pruned or prefill visual features differ')
        m=meta['prefill_logits'];raw=arrays['prefill_logits'].tobytes()
        if m['dtype']=='torch.bfloat16':v=(np.frombuffer(raw,dtype='<u2').astype(np.uint32)<<16).view(np.float32)
        elif m['dtype'] in ('torch.float16','torch.float32','torch.float64'):
            v=np.frombuffer(raw,dtype={'torch.float16':'<f2','torch.float32':'<f4','torch.float64':'<f8'}[m['dtype']])
        else:raise ValueError('invalid logits dtype')
        if not v.size or not np.isfinite(v).all():raise ValueError('nonfinite logits')

def verify(records,journal,anchors,refs,tokenizer,run):
    if len(anchors)!=3 or len(records)!=8 or len(journal)!=8:raise ValueError('wrong B inventory')
    keys=('task','qid','group_id','mode','forced','output_ids','eos_ids','generation_scalars','min_new_tokens')
    if any({k:r[k] for k in keys}!=j for r,j in zip(records,journal)):
        raise ValueError('generation journal differs')
    expected=[]
    for i,row in enumerate(anchors):
        for forced in ((False,True) if i==0 else (False,)):
            for mode in ('C-DROP','C-SKIP'):expected.append((row['task'],str(row['qid']),row['group_id'],mode,forced))
    got=[(r['task'],str(r['qid']),r['group_id'],r['mode'],r['forced']) for r in records]
    if got!=expected:raise ValueError('unexpected B anchor/order')
    token=tokenizer.encode(' the',add_special_tokens=False)[0];natural={};checked=0
    for r in records:
        verify_arrays(r,run)
        validate_trace(r['decode_trace'],r['output_ids'],len(r['retained_ids']),int(r['adjusted_delta'][0][0]))
        if tokenizer.decode(r['output_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)!=r['answer']:
            raise ValueError('tokenizer output differs')
        checked+=1
    for i,row in enumerate(anchors):
        key=(row['task'],str(row['qid']))
        pair=[r for r in records if (r['task'],str(r['qid']))==key and not r['forced']]
        validate_pair(*pair,refs[key+(True,)],refs[key+(False,)])
        if i==0:
            forced=[r for r in records if (r['task'],str(r['qid']))==key and r['forced']]
            validate_forced_pair(*forced,*pair,token)
    return {'status':'passed','natural_pairs':3,'forced_pairs':1,'generations':checked,
            'GT_opened':False,'scoring_performed':False,'timing_claims':False}
