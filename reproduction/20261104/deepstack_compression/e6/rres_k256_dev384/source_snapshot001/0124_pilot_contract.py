"""Pure CPU contracts for a three-anchor natural-TTFT pilot; no GPU admission."""
from collections import Counter
import random

ARMS=('pixel_t0','pixel_t045','eadp_k256')
RECIPE={'do_sample':False,'max_new_tokens':2048,'top_p':0.001,'top_k':1,
        'temperature':0.01,'repetition_penalty':1.0,'num_beams':1,'use_cache':True}

def make_schedule(anchors,seed=20261105):
    if len(anchors)!=3 or len({a['task'] for a in anchors})!=3 or len({a['group_id'] for a in anchors})!=3:
        raise ValueError('three distinct task/image-group anchors required')
    if any(a.get('split')!='DEV' for a in anchors):raise ValueError('only DEV')
    rng=random.Random(seed); tasks=list(anchors);rng.shuffle(tasks); jobs=[]
    for block,a in enumerate(tasks):
        arms=list(ARMS);rng.shuffle(arms)
        for arm in arms:
            jobs.append({'block':block,'order':len(jobs),'arm':arm,'task':a['task'],
                         'qid':str(a['qid']),'group_id':a['group_id']})
    return jobs

def validate_schedule(jobs,anchors,seed=20261105):
    if jobs!=make_schedule(anchors,seed):raise ValueError('schedule differs from frozen random order')
    if len(jobs)!=9 or Counter(j['arm'] for j in jobs)!=Counter({a:3 for a in ARMS}):raise ValueError('bad schedule')
    for block in range(3):
        group=jobs[block*3:block*3+3]
        if {j['arm'] for j in group}!=set(ARMS) or len({(j['task'],j['qid'],j['group_id']) for j in group})!=1:raise ValueError('bad paired block')
    return True

def validate_generation(tokens,eos_ids,first_length,prompt_length,prompt_in_return):
    if not isinstance(tokens,list) or not 1<=len(tokens)<=2048 or any(type(x) is not int for x in tokens):raise ValueError('invalid natural tokens')
    if not eos_ids or any(type(x) is not int for x in eos_ids):raise ValueError('missing EOS IDs')
    if first_length!=(prompt_length+1 if prompt_in_return else 1):raise ValueError('first token boundary length')
    eos=tokens[-1] in eos_ids;truncated=not eos and len(tokens)==2048
    if not eos and not truncated:raise ValueError('missing natural terminal reason')
    return {'eos':eos,'truncated':truncated,'output_tokens':len(tokens)}

def verify_expected(actual,expected):
    if not isinstance(expected,list) or not expected or actual!=expected:raise ValueError('output differs from accepted same-arm reference')
    return True
