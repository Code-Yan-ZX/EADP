"""No-GT full-panel checks; summaries do not prove tensor-byte reconstruction."""
def check_record(r, choice, e1, decode):
    for name in ('task','qid','group_id','message_sha256','image_sha256'):
        if str(r.get(name)) != str(choice.get(name)):
            raise ValueError('input identity '+name)
    for name in ('task','qid','group_id','message_sha256'):
        if str(r.get(name)) != str(e1.get(name)):
            raise ValueError('E1 input identity '+name)
    if (r.get('seed') != choice.get('seed') or r.get('derived_seed64') != choice.get('derived_seed64') or
        r.get('selection_mode') != 'e6_random_fixed_seed' or r.get('budget') != 256 or r.get('ds_on') is not True or
        r.get('greedy_calls') != 1 or r.get('actual_injection_verified') is not True or
        r.get('extra_author_greedy') is not True or r.get('timing_eligible') is not False or r.get('forced') is not False or r.get('protocol') != 'natural'):
        raise ValueError('method identity')
    ids = choice.get('spatial_ids')
    if not isinstance(ids,list) or len(ids)!=256 or ids!=sorted(set(ids)) or any(type(x) is not int or not 0<=x<1024 for x in ids) or r.get('ordered_ids')!=ids or r.get('spatial_ids')!=ids:
        raise ValueError('random selection')
    for actual, reference in (('processor','processor'),('main_visual','visual_sha256'),('full_deepstack','deepstack_full_sha256'),('full_position_ids','original_position_ids'),('full_delta','full_rope_delta')):
        if r.get(actual) != e1.get(reference): raise ValueError('E1 identity '+actual)
    start,n,pos = r['image_start'],r['original_length'],r['full_position_ids']
    ret = list(range(start))+[start+i for i in ids]+list(range(start+1024,n))
    if r.get('retained_ids')!=ret or len(pos)!=3 or any(len(a)!=1 or len(a[0])!=n for a in pos) or r.get('prefill_position_ids')!=[[[a[0][i] for i in ret]] for a in pos]:
        raise ValueError('P3 retained position')
    delta = max(max(a[0]) for a in pos)+1-n
    if r.get('full_delta') != [[delta]] or r.get('adjusted_delta') != [[delta+768]]:
        raise ValueError('P3 delta')
    mask = [False]*len(ret); mask[start:start+256]=[True]*256
    if r.get('injected_mask') != [mask] or len(r.get('full_deepstack',[])) != 3 or len(r.get('selected_deepstack',[])) != 3 or r.get('selected_deepstack') != r.get('injected_deepstack'):
        raise ValueError('DS inventory or selected/injected mismatch')
    if r.get('arrays',{}).get('kind') != 'summary_only': raise ValueError('retention')
    outputs,eos,trace = r.get('output_ids'),r.get('eos_ids'),r.get('decode_trace')
    if not isinstance(outputs,list) or not outputs or any(type(x) is not int for x in outputs) or len(outputs)>2048 or eos!=e1['eos_ids'] or r.get('generation_scalars')!=e1['generation_scalars']:
        raise ValueError('generation recipe/outputs')
    ended = outputs[-1] in eos
    if r.get('eos') is not ended or r.get('truncated') is not (not ended and len(outputs)==2048) or not (ended or len(outputs)==2048) or r.get('answer') != decode(outputs):
        raise ValueError('termination or decoded answer')
    if not isinstance(trace,list) or len(trace)!=len(outputs): raise ValueError('decode count')
    for i,t in enumerate(trace):
        prev = 0 if i==0 else len(ret)+i-1; after = len(ret)+i
        position = r['prefill_position_ids'] if i==0 else [[[prev+delta+768]]]*3
        if t.get('position_ids')!=position or t.get('cache_position')!=(list(range(len(ret))) if i==0 else [prev]) or t.get('rope_delta')!=[[delta+768]] or (t.get('layer_cache_lengths') not in ([],[0]*36) if i==0 else t.get('layer_cache_lengths')!=[prev]*36) or t.get('layer_cache_lengths_after')!=[after]*36:
            raise ValueError('decode cache/position')
    return True

def inventory(records, generated, journal, schedule):
    if len(schedule)!=1152 or len(records)!=1152 or len(generated)!=1152 or len(journal)!=2304:
        raise ValueError('incomplete full panel')
    def key(x): return (x['task'],str(x['qid']),x['seed'])
    for i,(row,seed) in enumerate(schedule):
        expected=(row['task'],str(row['qid']),seed); r,g=records[i],generated[i]
        if key(r)!=expected or key(g['scheduled'])!=expected or key(g['raw'])!=expected:
            raise ValueError('arm order')
        if r['output_ids']!=g['raw'].get('output_ids') or r['eos_ids']!=g['raw'].get('eos_ids') or r['generation_scalars']!=g['raw'].get('generation_scalars'):
            raise ValueError('original generated output mismatch')
        a,b=journal[2*i:2*i+2]
        if a.get('event')!='started' or b.get('event')!='recorded' or key(a)!=expected or key(b)!=expected:
            raise ValueError('journal order or failed arm')
    return True
