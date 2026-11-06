"""CPU validation and conservative stage summaries for sanitized NVML windows."""
MIB=1024*1024

def summarize(samples,target_pid,uuid,bounds,max_gap_ns):
    names=('release','model_load_start','model_load_end','request_start','request_end','teardown_end')
    if any(type(bounds.get(k)) is not int or bounds[k]<=0 for k in names): raise ValueError('invalid boundary')
    t=[bounds[k] for k in names]
    if not (t[0]<t[1]<t[2]<=t[3]<t[4]<=t[5]): raise ValueError('boundary order')
    if type(target_pid) is not int or target_pid<=0 or not isinstance(uuid,str) or not uuid or type(max_gap_ns) is not int or max_gap_ns<=0 or not samples:
        raise ValueError('invalid identity or gap')
    rows=[]; prev=None; prev_start=None; gaps=[]; spans=[]; widths=[]
    for s in samples:
        a,b=s.get('query_start_ns'),s.get('query_end_ns')
        if s.get('status')!='completed' or s.get('gpu_uuid')!=uuid or type(a) is not int or type(b) is not int or not 0<a<=b:
            raise ValueError('failed or malformed sample')
        if prev is not None:
            if a<prev: raise ValueError('query overlap/reorder')
            gaps.append(a-prev)
            spans.append(b-prev_start)
        widths.append(b-a)
        if b-a>max_gap_ns or (spans and spans[-1]>max_gap_ns): raise ValueError('query duration or conservative sample spacing exceeds bound')
        mem=s.get('device_memory_bytes'); procs=s.get('processes')
        if not isinstance(mem,dict) or set(mem)!={'total','used','free','reserved'} or not isinstance(procs,list): raise ValueError('missing inventory')
        def metric(x):
            if x is not None and (type(x) is not int or x<0 or x%MIB): raise ValueError('invalid MiB metric')
            return x
        for v in mem.values(): metric(v)
        pids=set(); target=None; graphics=[]
        for p in procs:
            pid,kind=p.get('pid'),p.get('type')
            if type(pid) is not int or pid<=0 or pid in pids or kind not in ('C','G','C+G') or 'used_bytes' not in p: raise ValueError('invalid process inventory')
            pids.add(pid); metric(p['used_bytes'])
            if pid==target_pid and kind in ('C','C+G'): target=p
            elif kind in ('C','C+G'): raise ValueError('foreign compute process observed')
            else: graphics.append(pid)
        rows.append({'target_present':target is not None,'target_used_bytes':None if target is None else target['used_bytes'], 'device':mem,'graphics':graphics})
        prev=b; prev_start=a
    if samples[0]['query_end_ns']>t[0] or samples[-1]['query_end_ns']<t[-1]: raise ValueError('missing baseline or final boundary coverage')
    windows={'pre_cuda':(None,t[0]),'model_load':(t[1],t[2]),'request':(t[3],t[4]),'teardown':(t[4],t[5])}
    result={}
    for name,(start,end) in windows.items():
        overlap=[i for i,s in enumerate(samples) if (start is None or s['query_end_ns']>=start) and s['query_start_ns']<=end]
        inside=[i for i in overlap if (start is None or samples[i]['query_start_ns']>=start) and samples[i]['query_end_ns']<=end]
        vals={k:[] for k in ('target','total','used','free','reserved')}
        for i in overlap:
            vals['target'].append(rows[i]['target_used_bytes'])
            for k in ('total','used','free','reserved'): vals[k].append(rows[i]['device'][k])
        metrics={k:{'known':sum(v is not None for v in vv),'unknown':sum(v is None for v in vv),'max_known_bytes':max((v for v in vv if v is not None),default=None)} for k,vv in vals.items()}
        result[name]={'overlap_indices':overlap,'fully_contained_indices':inside,'overlap_metrics':metrics,'target_absent':sum(not rows[i]['target_present'] for i in overlap),'target_present_unknown':sum(rows[i]['target_present'] and rows[i]['target_used_bytes'] is None for i in overlap),'graphics_pids':sorted({p for i in overlap for p in rows[i]['graphics']})}
    return {'status':'summarized','stages':result,'sample_count':len(samples),'max_unsampled_gap_ns':max(gaps,default=None),'max_possible_sample_spacing_ns':max(spans,default=None),'max_query_duration_ns':max(widths), 'limits':['unknown or absent target is not zero','overlap maxima can include measurements outside the named stage; query windows do not localize sample time','sampled MiB maxima can miss transients','device usage and process usage are distinct','graphics observations do not prove absence of unobserved interference']}

if __name__=='__main__': raise SystemExit('CPU contract only: no GPU authority or sampler')
