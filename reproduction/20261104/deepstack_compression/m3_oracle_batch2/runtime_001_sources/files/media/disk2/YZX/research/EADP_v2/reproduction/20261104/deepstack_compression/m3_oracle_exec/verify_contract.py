"""No-GT independent record-level replay of one completed oracle question.

Tensor hashes and runtime self-checks are evidence, not reconstruction of the
unarchived full feature/gradient tensors. No accuracy score is computed.
"""
import math
from collections import Counter
import search
import importlib.util
from pathlib import Path
_decode_spec=importlib.util.spec_from_file_location('m3_oracle_verify_decode',Path(__file__).resolve().parent.parent/'e4/decode_observer.py')
_decode=importlib.util.module_from_spec(_decode_spec);_decode_spec.loader.exec_module(_decode)
validate_trace=_decode.validate_trace

def validate(row,e1,events,decode):
    counts=Counter(x['event'] for x in events)
    singles=('e1_baseline_first_logits','e1_baseline_verified','reference_access_started','target_encoded',
             'full_gradient_completed','question_started','search_completed','question_completed')
    if any(counts[x]!=1 for x in singles) or any(counts[x]!=6 for x in ('generation_started','generation_returned','generation_completed')):
        raise ValueError('question event inventory incomplete')
    if counts['evaluation_started']!=counts['evaluation_completed'] or not 1<=counts['evaluation_completed']<=49:
        raise ValueError('teacher evaluation count invalid')
    allowed=set(singles)|{'evaluation_started','evaluation_completed','generation_started','generation_returned','generation_completed'}
    if set(counts)!=allowed or events[-1]['event']!='question_completed':raise ValueError('question event set invalid')
    def one(name):return next(x for x in events if x['event']==name)
    sequence=[x['event'] for x in events]
    if not (sequence.index('e1_baseline_first_logits')<sequence.index('e1_baseline_verified')<
            sequence.index('reference_access_started')<sequence.index('target_encoded')<sequence.index('full_gradient_completed')<
            sequence.index('question_started')<sequence.index('search_completed')<sequence.index('generation_started')):
        raise ValueError('baseline/reference/search sequence invalid')
    first=one('e1_baseline_first_logits')
    if first.get('identical') is not True or first.get('actual')!=first.get('expected') or first.get('actual')!=e1['logits'][0]:
        raise ValueError('baseline actual first logits mismatch')
    binding=one('e1_baseline_verified');start=one('question_started')
    if (binding.get('qid')!=str(row['qid']) or binding.get('arm')!='P3-D1' or binding.get('verified') is not True or
        start.get('baseline')!={k:v for k,v in binding.items() if k!='event'}):raise ValueError('baseline binding mismatch')
    gradient=one('full_gradient_completed');util=gradient['utility'];g=gradient['evidence']
    if (gradient.get('gradient_streams')!=4 or not math.isfinite(gradient.get('mean_nll',float('nan'))) or
        len(util)!=4 or any(len(x)!=1024 or any(not math.isfinite(float(y)) for y in x) for x in util) or
        g.get('features')!=[e1['visual_sha256']]+e1['deepstack_full_sha256'] or len(g.get('gradients',[]))!=4 or
        any(x.get('shape')!=[1024,4096] or not isinstance(x.get('sha256'),str) or len(x['sha256'])!=64 for x in g['gradients']) or
        len(g.get('gradient_nonzero_counts',[]))!=4 or any(not isinstance(x,int) or x<=0 for x in g['gradient_nonzero_counts'])):
        raise ValueError('full gradient evidence invalid')
    proposal=search.p.proposals(str(row['qid']),e1['ordered_ids'],util)
    if start['proposal']!=proposal:raise ValueError('proposal replay mismatch')
    pending=None;losses={}
    for event in events:
        if event['event']=='evaluation_started':
            if pending is not None:raise ValueError('overlapping evaluations')
            pending=tuple(event['spatial_ids'])
            if pending in losses:raise ValueError('duplicate uncached evaluation')
        elif event['event']=='evaluation_completed':
            key=tuple(event['spatial_ids']);value=event['mean_nll']
            if key!=pending or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('evaluation journal mismatch')
            losses[key]=value;pending=None
    if pending is not None:raise ValueError('unfinished evaluation')
    baseline=tuple(sorted(e1['ordered_ids']));expected_keys={baseline};winners=[]
    if baseline not in losses:raise ValueError('missing baseline teacher loss')
    for r in (8,16):
        block=proposal['by_r'][str(r)];n=block['n_proposals_per_source']
        for source in search.p.SOURCES:
            best=(losses[baseline],-1,list(baseline))
            for i,added in enumerate(block['sources'][source]):
                ids=search.p.replacement(e1['ordered_ids'],added,r);key=tuple(ids);expected_keys.add(key)
                if key not in losses:raise ValueError('missing proposed teacher loss')
                candidate=(losses[key],i,ids)
                if candidate[:2]<best[:2]:best=candidate
            winners.append({'r':r,'source':source,'proposal_index':best[1],'spatial_ids':best[2],'mean_nll':best[0],'n_proposals':n})
    finish=one('search_completed')
    if set(losses)!=expected_keys or finish['unique_evaluations']!=len(losses) or finish['winners']!=winners or finish['baseline_nll']!=losses[baseline]:
        raise ValueError('teacher search/winner replay mismatch')
    gens=[e for e in events if e['event'] in ('generation_started','generation_returned','generation_completed')]
    if [x['event'] for x in gens]!=['generation_started','generation_returned','generation_completed']*6:raise ValueError('generation exposure order')
    for winner,idx in zip(winners,range(0,len(gens),3)):
        a,b,c=gens[idx:idx+3];identity={k:winner[k] for k in ('source','r','spatial_ids')}
        if any(any(x.get(k)!=v for k,v in identity.items()) for x in (a,b,c)):raise ValueError('generation winner mismatch')
        raw=b['output'];output=c['output'];ids=output['token_ids'];eos=bool(ids and ids[-1] in e1['eos_ids']);truncated=not eos and len(ids)>=2048
        if (raw['token_ids']!=ids or raw['eos_ids']!=e1['eos_ids'] or raw['generation_scalars']!=e1['generation_scalars'] or
            output['eos_ids']!=e1['eos_ids'] or output['generation_scalars']!=e1['generation_scalars'] or output['eos']!=eos or
            output['truncated']!=truncated or not (eos or truncated) or len(ids)>2048 or output['answer']!=decode(ids) or
            output.get('actual_selection_verified') is not True):raise ValueError('generated output/recipe/decoding mismatch')
        evidence=output['evidence'];spatial=winner['spatial_ids'];startpos=e1['image_start'];length=e1['original_length']
        retained=list(range(startpos))+[startpos+i for i in spatial]+list(range(startpos+1024,length))
        positions=[[[axis[0][i] for i in retained]] for axis in e1['original_position_ids']]
        mask=[[False]*startpos+[True]*256+[False]*(length-startpos-1024)]
        validate_trace(evidence['decode_trace'],ids,len(retained),e1['full_rope_delta'][0][0]+768)
        if (evidence['processor']!=e1['processor'] or evidence['full_streams']!=[e1['visual_sha256']]+e1['deepstack_full_sha256'] or
            evidence['spatial_ids']!=spatial or evidence['retained_ids']!=retained or evidence['position_ids']!=positions or
            evidence['rope_delta']!=[[e1['full_rope_delta'][0][0]+768]] or evidence['ds_mask']!=mask or len(evidence['selected_ds'])!=3):
            raise ValueError('actual P3/DeepStack generation evidence mismatch')
    if one('question_completed').get('generation_count')!=6:raise ValueError('generation denominator mismatch')
    return {'qid':str(row['qid']),'evaluations':len(losses),'new_generations':6,'baseline_record_sha256':binding['record_sha256']}
