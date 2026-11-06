"""Pre-review pure timing statistics; no file, GT or model access."""
from collections import Counter
import math
import numpy as np

TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')
ARMS=('eadp','rres')
CHECKS=('tokens_match','response_match','processor_match','image_match','first_criteria_length_match','criteria_count_match')
ENDPOINTS={'ttft_ms':'first_ns','generation_ms':'generate_end_ns','response_ms':'response_end_ns'}


def validate_and_collect(schedule,process_records):
    measured=schedule['measured_inputs'];warmups=schedule['warmup_inputs'];processes=schedule['processes']
    if len(measured)!=32 or len(warmups)!=30 or len(processes)!=12 or len(process_records)!=12:
        raise ValueError('full sample/process denominator')
    def key(row):return (row['task'],str(row['qid']),row['group_id'])
    if (len({key(x) for x in measured})!=32 or len({key(x) for x in warmups})!=30 or
        len({x['group_id'] for x in measured+warmups})!=62 or
        Counter(x['task'] for x in measured)!=dict(zip(TASKS,(11,11,10))) or
        Counter(x['task'] for x in warmups)!={x:10 for x in TASKS}):raise ValueError('cohort/group identity')
    expected_measured=Counter((key(x),arm,rep) for x in measured for arm in ARMS for rep in range(5))
    expected_warm=Counter((key(x),arm,0) for x in warmups for arm in ARMS)
    actual_measured=Counter();actual_warm=Counter();values={};rows=[];all_counts=Counter();block_assignment={}
    if [p['order'] for p in processes]!=list(range(12)):raise ValueError('process order')
    process_signatures=[]
    for process,records in zip(processes,process_records):
        task=process['task'];arm=process['arm'];block=process['block'];requests=process['requests']
        if task not in TASKS or arm not in ARMS or block not in (0,1) or len(records)!=len(requests):raise ValueError('process schema')
        process_signatures.append((task,arm,block))
        if [x['phase'] for x in requests[:5]]!=['warmup']*5 or any(x['phase']!='measured' for x in requests[5:]):raise ValueError('warmup order')
        for index,(expected,record) in enumerate(zip(requests,records)):
            if (record.get('event')!='completed' or record.get('index')!=index or record.get('arm')!=arm or
                record.get('request')!=expected or expected.get('task')!=task):raise ValueError('request/order identity')
            checks=record.get('checks')
            if not isinstance(checks,dict) or any(checks.get(k) is not True for k in CHECKS):raise ValueError('identity check failed')
            t=record.get('times',{});names=('t0_ns','first_ns','generate_end_ns','response_end_ns')
            if (any(type(t.get(k)) is not int or t[k]<=0 for k in names) or
                not t['t0_ns']<t['first_ns']<=t['generate_end_ns']<=t['response_end_ns'] or
                t.get('raw_persisted_after_all_endpoints') is not True):raise ValueError('timing boundary')
            output=record.get('output_ids');conf=record.get('generation',{}).get('effective_generation_config',{})
            eos=conf.get('eos_token_id');eos=[eos] if type(eos) is int else eos
            if not isinstance(output,list) or not output or any(type(x) is not int for x in output) or not isinstance(eos,list) or not eos or any(type(x) is not int for x in eos):raise ValueError('output/EOS')
            if type(t.get('criteria_calls')) is not int or t['criteria_calls']!=len(output):raise ValueError('token callback count')
            if output[-1] not in eos and len(output)!=2048:raise ValueError('incomplete natural generation')
            phase=expected['phase'];rep=expected['replicate'];k=key(expected)
            if block_assignment.setdefault(k,block)!=block:raise ValueError('paired image assigned to different blocks')
            if phase=='measured' and type(rep) is int and 0<=rep<5:actual_measured[(k,arm,rep)]+=1
            elif phase=='warmup' and rep==0:actual_warm[(k,arm,rep)]+=1
            else:raise ValueError('phase/replicate')
            durations={name:(t[endpoint]-t['t0_ns'])/1_000_000 for name,endpoint in ENDPOINTS.items()}
            row={'process':process['order'],'request_index':index,'task':task,'qid':str(expected['qid']),'group_id':expected['group_id'],'arm':arm,'block':block,'phase':phase,'replicate':rep,
                 'times':{k:t[k] for k in names},'durations_ms':durations,'output_tokens':len(output),'eos':output[-1] in eos,
                 'truncated':output[-1] not in eos and len(output)==2048,'checks':checks}
            rows.append(row);all_counts[phase]+=1
            if phase=='measured':values[(k,arm,rep)]=durations
    if Counter(process_signatures)!=Counter((t,a,b) for t in TASKS for a in ARMS for b in (0,1)):
        raise ValueError('process signatures')
    if actual_measured!=expected_measured or actual_warm!=expected_warm or all_counts!={'warmup':60,'measured':320}:
        raise ValueError('missing/duplicate samples')
    return values,rows


def summarize(schedule,process_records,bootstrap=10000,seed=20261106):
    if bootstrap!=10000 or seed!=20261106:raise ValueError('fixed bootstrap plan')
    values,rows=validate_and_collect(schedule,process_records)
    result={'tasks':{},'per_image':[],'per_job':rows,'counts':{'images':32,'processes':12,'warmups':60,'measured':320},
            'bootstrap':{'seed':seed,'draws':bootstrap,'generator':'numpy.random.default_rng','percentile':'linear','numpy_version':np.__version__}}
    rng=np.random.default_rng(seed)
    for task in TASKS:
        images=[x for x in schedule['measured_inputs'] if x['task']==task]
        draws=rng.integers(0,len(images),size=(bootstrap,len(images)))
        table={m:{a:[] for a in ARMS} for m in ENDPOINTS}
        for image in images:
            k=(task,str(image['qid']),image['group_id']);entry={'task':task,'qid':str(image['qid']),'group_id':image['group_id'],'metrics':{}}
            for metric in ENDPOINTS:
                for arm in ARMS:table[metric][arm].append(float(np.median([values[(k,arm,r)][metric] for r in range(5)])))
                e=table[metric]['eadp'][-1];rr=table[metric]['rres'][-1]
                if not math.isfinite(e) or not math.isfinite(rr) or e<=0 or rr<=0:raise ValueError('invalid image median')
                entry['metrics'][metric]={'eadp_median_ms':e,'rres_median_ms':rr,'difference_ms':rr-e,'ratio':rr/e}
            result['per_image'].append(entry)
        task_result={'images':len(images),'metrics':{}}
        for metric,arms in table.items():
            e=np.array(arms['eadp'],dtype=np.float64);rr=np.array(arms['rres'],dtype=np.float64);difference=rr-e;ratio=rr/e
            db=np.median(difference[draws],axis=1);rb=np.median(ratio[draws],axis=1)
            task_result['metrics'][metric]={'eadp_median_ms':float(np.median(e)),'eadp_range_ms':[float(e.min()),float(e.max())],
                'rres_median_ms':float(np.median(rr)),'rres_range_ms':[float(rr.min()),float(rr.max())],
                'paired_difference_median_ms':float(np.median(difference)), 'difference_ci95_ms':np.percentile(db,[2.5,97.5],method='linear').tolist(),
                'paired_ratio_median':float(np.median(ratio)),'ratio_ci95':np.percentile(rb,[2.5,97.5],method='linear').tolist()}
        result['tasks'][task]=task_result
    return result
