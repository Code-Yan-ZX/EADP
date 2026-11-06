"""Separate component profiler primitives; no model or GPU admission authority.

Callers must independently prove component inputs and actual model identities,
and must never reuse these synchronized event measurements as end-to-end TTFT.
"""
import json,math,time

def measure_component(operation,event_factory,clock=time.perf_counter_ns):
    """CUDA interval plus host wall time including this profiler's boundary sync."""
    begin=event_factory();end=event_factory();start=clock();begin.record();output=operation();end.record();end.synchronize();finish=clock()
    value=float(begin.elapsed_time(end))
    if not math.isfinite(value) or value<0:raise ValueError('invalid profiler event duration')
    if finish<start:raise ValueError('nonmonotonic profiler clock')
    return {'gpu_event_ms':value,'host_wall_ms':(finish-start)/1e6}


def pilot_schedule(full):
    """First frozen image only; 60 component jobs, no post-hoc choice."""
    if full.get('stage','full')!='full' or len(full.get('images',[]))!=32 or len(full.get('jobs',[]))!=1920:raise ValueError('full schedule required')
    image=full['images'][0];jobs=full['jobs'][:60]
    if any((x['task'],x['group_id'])!=tuple(image) or x['order']!=i for i,x in enumerate(jobs)):raise ValueError('first image schedule not contiguous')
    return {**full,'stage':'pilot','images':[image],'jobs':jobs}

def replay(schedule,expected,prepare,event_factory,synchronize,journal,warmup=3):
    """Replay frozen order with one image's feature references resident at once.

    `prepare(task,group,component)` returns (operation, identity). Real callers
    must independently verify each identity against its accepted input/features.
    """
    if json.dumps(schedule,sort_keys=True)!=json.dumps(expected,sort_keys=True) or schedule.get('scope')!='separate batch1 DEV profiler only; not end-to-end TTFT':raise ValueError('unapproved component schedule')
    jobs=schedule.get('jobs');images=schedule.get('images');components=schedule.get('components')
    stage=schedule.get('stage','full');image_count={'pilot':1,'full':32}.get(stage)
    if (image_count is None or not isinstance(jobs,list) or len(jobs)!=image_count*60 or len(images)!=image_count or len({tuple(x) for x in images})!=image_count or
        components!=['selector','ds_merger_three_streams','ds_injection'] or schedule.get('repeat_per_image_component')!=20 or type(warmup) is not int or warmup<1):raise ValueError('invalid microbenchmark inventory')
    cache={};current=None;count=0
    for index,job in enumerate(jobs):
        if job.get('order')!=index or job.get('component') not in components:raise ValueError('invalid job order')
        group=(job['task'],job['group_id']);component=job['component']
        if group!=current:cache.clear();current=group
        if component not in cache:
            operation,identity=prepare(*group,component)
            if not callable(operation) or not isinstance(identity,dict) or identity.get('task')!=group[0] or identity.get('group_id')!=group[1] or identity.get('component')!=component:raise ValueError('component input identity missing')
            for _ in range(warmup):operation()
            synchronize();cache[component]=(operation,identity)
        operation,identity=cache[component]
        try:elapsed=measure_component(operation,event_factory)
        except Exception as exc:
            journal({'status':'failed','job':job,'error_type':type(exc).__name__,'error':str(exc)});raise
        journal({'status':'measured','job':job,**elapsed,'input_identity':identity,'scope':'separate component profiler; not TTFT'})
        count+=1
    return {'status':'completed','measurements':count,'warmups_per_component_image':warmup}
