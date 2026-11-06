"""CPU content checks for E2/E3 diagnostic draft; not a launch acceptance gate.

Complete acceptance additionally requires an independently reviewed gate for
source/weight/input identity, supervisor and actual Docker exit. No score/GT.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from local_imports import load_local
validate_one=load_local('verify_interventions').validate_one

ARMS=(('E2','000'),('E2','111'),('E2','100'),('E2','010'),('E2','001'),('E3','PERM'),('E3','MEAN'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def lines(path):
    b=Path(path).read_bytes()
    if b and not b.endswith(b'\n'):raise ValueError('partial JSONL')
    return [json.loads(x) for x in b.splitlines()]

def validate_kv_reference(intervention,control):
    common=('processor','ordered_ids','spatial_ids','retained_ids','main_visual','pruned_visual','full_deepstack',
            'full_position_ids','full_delta','prefill_position_ids','adjusted_delta','prefill_embeds','prefill_mask','prefill_logits','output_ids')
    if any(intervention[x]!=control[x] for x in common) or intervention['cache_fingerprints']!=control['cache_fingerprints']:
        raise ValueError('forced E2/reference prefill or 36-layer KV content differs')

def validate_e1_endpoint(record,endpoint):
    if any(record[x]!=endpoint[x] for x in ('output_ids','answer','eos_ids','eos','truncated')):
        raise ValueError('E1 endpoint differs')
    logits=endpoint.get('logits')
    if not isinstance(logits,list) or len(logits)!=1 or record['prefill_logits']!=logits[0]:
        raise ValueError('E1 prefill logits differ')

def validate_content(run,panel,e1_records,tokenizer):
    """Raise on content mismatch; caller must also check provenance and exit."""
    run=Path(run).resolve();records=lines(run/'records.jsonl');journal=lines(run/'generated.jsonl')
    expected=[]
    for x in panel:
        expected += [(x['task'],str(x['qid']),family,arm,forced) for family,arm in ARMS for forced in (False,True)]
        expected += [(x['task'],str(x['qid']),'REF',arm,True) for arm in ('D0','D1')]
    inventory=[(r['task'],str(r['qid']),r['family'],r['arm'],r['forced']) for r in records]
    if inventory!=expected or len(journal)!=len(records) or len(panel)!=3:raise ValueError('diagnostic inventory')
    refs={(r['task'],str(r['qid']),r['arm']):r for r in e1_records if r.get('forced') is False}
    original={(r['task'],str(r['qid'])):r for r in panel};natural={};forced_rows={}
    forced_token=tokenizer.encode(' the',add_special_tokens=False)[0]
    fields=('task','qid','group_id','budget','ds_on','output_ids','eos_ids','generation_scalars','forced','family','arm')
    same=('processor','ordered_ids','spatial_ids','retained_ids','main_visual','pruned_visual','full_deepstack',
          'selected_deepstack','injected_mask','injected_deepstack','full_position_ids','full_delta',
          'prefill_position_ids','adjusted_delta','prefill_embeds','prefill_mask','prefill_logits','intervention_metadata')
    for r,g in zip(records,journal):
        item=original[(r['task'],str(r['qid']))];key=(r['task'],str(r['qid']),r['family'],r['arm'])
        if any(r[x]!=item[x] for x in ('task','qid','group_id','image_sha256','message_sha256')):raise ValueError('input identity')
        if g!={k:r[k] for k in fields}:raise ValueError('immediate generation journal')
        ids=r['output_ids'];isforced=r['forced']
        if isforced:
            if ids!=[forced_token]*33 or r['eos_ids']!=[] or r['forced_token']!=forced_token or r['eos'] or r['truncated']:
                raise ValueError('forced prefix')
            if r['family']!='REF' and (key not in natural or any(r[k]!=natural[key][k] for k in same)):raise ValueError('natural/forced prefill mismatch')
            forced_rows[key]=r
        else:
            end=bool(ids and ids[-1] in r['eos_ids']);trunc=bool(len(ids)==2048 and not end)
            if not ids or len(ids)>2048 or r['eos']!=end or r['truncated']!=trunc or not (end or trunc):raise ValueError('natural endpoint')
            natural[key]=r
        if tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)!=r['answer']:raise ValueError('token decode')
        ref=refs[(r['task'],str(r['qid']),'P3-D0' if r['family']=='REF' and r['arm']=='D0' else 'P3-D1')]
        scalars=dict(ref['generation_scalars']);scalars['max_new_tokens']=33 if isforced else 2048
        if r['generation_scalars']!=scalars:raise ValueError('generation parameters')
        mapping={'processor':'processor','ordered_ids':'ordered_ids','spatial_ids':'spatial_ids','retained_ids':'retained_ids',
                 'main_visual':'visual_sha256','pruned_visual':'pruned_sha256','full_deepstack':'deepstack_full_sha256',
                 'prefill_position_ids':'prefill_position_ids','adjusted_delta':'rope_delta','prefill_embeds':'prefill_embed_sha256','prefill_mask':'attention_mask_sha256'}
        if any(r[a]!=ref[b] for a,b in mapping.items()):raise ValueError('E1 input/selection/prefill identity')
        if not isforced and (r['family'],r['arm']) in (('E2','000'),('E2','111')):
            endpoint=refs[(r['task'],str(r['qid']),'P3-D0' if r['arm']=='000' else 'P3-D1')]
            validate_e1_endpoint(r,endpoint)
        ar=r['arrays'];path=Path(ar['path']).resolve()
        if ar['kind']!='full_arrays' or path.parent!=run/'arrays' or sha(path)!=ar['sha256']:raise ValueError('array evidence')
        with np.load(path,allow_pickle=False) as a:
            validate_one(r,a)
            logits=a['prefill_logits'];meta=r['prefill_logits']
            if hashlib.sha256(logits.tobytes()).hexdigest()!=meta['sha256']:raise ValueError('logits hash')
            bits=logits.tobytes();dtype=meta['dtype']
            finite=np.all((np.frombuffer(bits,dtype=np.uint16)&0x7f80)!=0x7f80) if dtype=='torch.bfloat16' else np.all(np.isfinite(np.frombuffer(bits,dtype={'torch.float16':np.float16,'torch.float32':np.float32,'torch.float64':np.float64}[dtype])))
            if not finite:raise ValueError('nonfinite logits')
            if isforced:
                for k in ('input_ids','attention_mask','pixel_values','image_grid_thw'):
                    if hashlib.sha256(a['processor_'+k].tobytes()).hexdigest()!=r['processor'][k]['sha256']:raise ValueError('processor arrays')
    for item in panel:
        key=(item['task'],str(item['qid']))
        for e2,reference in (('000','D0'),('111','D1')):
            intervention=forced_rows[(*key,'E2',e2)];control=forced_rows[(*key,'REF',reference)]
            validate_kv_reference(intervention,control)
    return {'natural_verified':len(natural),'forced_verified':len(records)-len(natural),'forced_kv_reference_pairs':6,'kv_fingerprint_steps':33,'GT_opened':False,'scoring_performed':False}
