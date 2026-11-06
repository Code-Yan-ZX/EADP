"""Exploratory per-image CUDA-event summaries; no GPU execution or GT."""
import math, statistics

METHODS=('author','fused','graph')

def summarize(events, image_count=32, budgets=(128,256,512)):
    cases={}
    setup={}
    for event in events:
        if event.get('status')=='graph_setup':
            k=event['budget']
            if k in setup or event['result']['status']!='passed': raise ValueError('bad setup')
            setup[k]=event['result']
        if event.get('status')!='case_completed': continue
        key=(event['image_index'],event['budget'])
        if key in cases or key[0] not in range(image_count) or key[1] not in budgets: raise ValueError('bad case key')
        result=event['result']
        if result.get('status')!='measured': raise ValueError('case unavailable')
        vals={}
        for method in METHODS:
            rows=[m for m in result['measurements'] if m['implementation']==method]
            if len(rows)!=5 or sorted(m['repeat'] for m in rows)!=list(range(5)): raise ValueError('wrong repeats')
            observed=[m for m in rows if m['status']=='measured']
            allowed=(method=='author' or result['candidate_matches_author'][method] is True)
            if not allowed:
                if observed: raise ValueError('failed candidate measured')
                vals[method]=None
                continue
            if len(observed)!=5 or any(m.get('equals_author') is not True for m in observed): raise ValueError('unverified measurement')
            times=[m['gpu_event_ms'] for m in observed]
            if any(not isinstance(t,(int,float)) or not math.isfinite(t) or t<=0 for t in times): raise ValueError('bad time')
            vals[method]=statistics.median(times)
        cases[key]=vals
    if set(cases)!={(i,k) for i in range(image_count) for k in budgets} or set(setup)!=set(budgets): raise ValueError('incomplete panel or setup')
    def distribution(values):
        if not values:return {'n':0}
        return {'n':len(values),'median_ms':statistics.median(values),'min_ms':min(values),'max_ms':max(values)}
    by_budget={}
    for k in budgets:
        paired=[i for i in range(image_count) if all(cases[i,k][m] is not None for m in METHODS)]
        by_budget[str(k)]={'all_available':{m:distribution([cases[i,k][m] for i in range(image_count) if cases[i,k][m] is not None]) for m in METHODS},'paired_image_indices':paired,'paired':{m:distribution([cases[i,k][m] for i in paired]) for m in METHODS},'excluded_image_indices':[i for i in range(image_count) if i not in paired],'per_image':{str(i):cases[i,k] for i in range(image_count)},'graph_setup':setup[k]}
    return by_budget
