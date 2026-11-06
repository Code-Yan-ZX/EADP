"""Bind sanitized device/process observations to existing sampler windows."""
def verify(expected,metadata,samples):
    if not isinstance(metadata,list) or not isinstance(samples,list) or not samples or len(metadata)!=len(samples):
        raise ValueError('one metadata observation per sampler query is required')
    device_values={k:[] for k in ('total','used','free','reserved')};target_used=[];target_indices=[];graphics=[]
    for index,(meta,sample) in enumerate(zip(metadata,samples)):
        if (meta.get('status')!='completed' or meta.get('gpu_uuid')!=expected['gpu_uuid']
            or sample.get('gpu_uuid')!=expected['gpu_uuid'] or sample.get('host_pid')!=expected['host_pid']
            or any(type(meta.get(k)) is not int for k in ('query_start_ns','query_end_ns'))
            or not sample['query_start_ns']<=meta['query_start_ns']<=meta['query_end_ns']<=sample['query_end_ns']):
            raise ValueError('metadata/sampler time or identity mismatch')
        processes=meta.get('processes');memory=meta.get('device_memory_bytes')
        if not isinstance(processes,list) or not isinstance(memory,dict) or set(memory)!={'total','used','free','reserved'}:
            raise ValueError('metadata schema')
        for v in memory.values():
            if v is not None and (type(v) is not int or v<0 or v%1048576):raise ValueError('device byte value')
        for key,value in memory.items():device_values[key].append(value)
        seen=set();target=None;others=[];g=[]
        for p in processes:
            if (set(p)!={'pid','type','used_bytes'} or type(p['pid']) is not int or p['pid']<=0
                or p['pid'] in seen or p['type'] not in ('C','G','C+G')
                or (p['used_bytes'] is not None and (type(p['used_bytes']) is not int or p['used_bytes']<0 or p['used_bytes']%1048576))):raise ValueError('process metadata malformed')
            seen.add(p['pid'])
            if p['type'] in ('C','C+G'):
                if p['used_bytes'] is None:raise ValueError('compute bytes unavailable')
                if p['pid']==expected['host_pid']:target=p['used_bytes']
                else:others.append({'pid':p['pid'],'used_bytes':p['used_bytes']})
            if p['type'] in ('G','C+G') and p['pid']!=expected['host_pid']:g.append(p)
        if target!=sample.get('used_bytes') or others!=sample.get('other_gpu_processes'):
            raise ValueError('worker or other compute observations differ')
        if others:raise ValueError('other compute process during sampling')
        if target is not None:target_used.append(target);target_indices.append(index)
        graphics.append({'sample_monotonic_ns':sample['monotonic_ns'],'other_graphics_processes':g})
    if not target_indices:raise ValueError('worker process never observed')
    first,last=target_indices[0],target_indices[-1];target_set=set(target_indices)
    coverage={key:{'known':sum(v is not None for v in values),'unknown':sum(v is None for v in values),
                   'complete':all(v is not None for v in values)} for key,values in device_values.items()}
    known_used=[v for v in device_values['used'] if v is not None]
    known_total=[v for v in device_values['total'] if v is not None]
    return {'status':'passed','sample_count':len(samples),
            'device_field_coverage':coverage,
            'device_used_max_of_known_samples_bytes':max(known_used) if known_used else None,
            'device_total_known_values_bytes':sorted(set(known_total)),
            'worker_used_max_of_present_samples_bytes':max(target_used),
            'worker_presence':{'present':len(target_indices),'absent_before_first':first,
                               'absent_between_first_and_last':sum(i not in target_set for i in range(first,last+1)),
                               'absent_after_last':len(samples)-last-1},
            'graphics_observations':graphics,
            'limitations':['device used and device capacity are not worker process memory',
                           'values are MiB-rounded samples; transient peaks can be missed',
                           'graphics processes are recorded as background, not treated as an idle device',
                           'device maxima use only known samples; unknown device values remain unknown',
                           'worker absence is not a zero-byte observation']}

if __name__=='__main__':raise SystemExit('locked: no GPU authorization')
