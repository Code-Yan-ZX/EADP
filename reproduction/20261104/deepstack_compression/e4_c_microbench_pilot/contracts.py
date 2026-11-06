"""CPU-testable single-image replay with per-invocation full output checking.

Profiler intervals exclude output checking and journal I/O. This is a component
diagnostic, not end-to-end latency. This module grants no GPU authority.
"""
import math,time
COMPONENTS=('selector','ds_merger_three_streams','ds_injection')

def run_jobs(schedule,expected,operations,references,event_factory,synchronize,equal,fingerprint,journal,hooks_clean,is_inference_mode,stream_is_expected,clock=time.perf_counter_ns):
 if schedule!=expected or schedule.get('stage')!='pilot' or len(schedule.get('images',[]))!=1 or len(schedule.get('jobs',[]))!=60 or tuple(schedule.get('components',[]))!=COMPONENTS:raise ValueError('invalid frozen pilot schedule')
 if set(operations)!=set(COMPONENTS) or set(references)!=set(COMPONENTS):raise ValueError('missing component')
 if not hooks_clean():raise ValueError('diagnostic hooks remain')
 if not is_inference_mode():raise ValueError('inference mode required for component replay')
 if not stream_is_expected():raise ValueError('unexpected CUDA stream')
 fingerprints={k:fingerprint(v) for k,v in references.items()};journal({'status':'reference_fingerprints','sha256':fingerprints})
 image=tuple(schedule['images'][0]);seen={x:0 for x in COMPONENTS};warmed=set()
 for n,job in enumerate(schedule['jobs']):
  component=job.get('component')
  if component not in seen or job.get('order')!=n or (job.get('task'),job.get('group_id'))!=image:raise ValueError('job order or image differs')
  op=operations[component]
  if component not in warmed:
   for i in range(3):
    if not hooks_clean():raise ValueError('hook reappeared before warmup')
    value=op();synchronize()
    if not equal(value,references[component]) or fingerprint(value)!=fingerprints[component]:raise ValueError('warmup full output differs')
    journal({'status':'warmup_validated','component':component,'index':i,'full_output_equal':True,'output_sha256':fingerprints[component]})
   warmed.add(component)
  if not hooks_clean():raise ValueError('hook reappeared before measurement')
  if not is_inference_mode():raise ValueError('inference mode disabled during component replay')
  if not stream_is_expected():raise ValueError('CUDA stream changed before measurement')
  begin,end=event_factory(),event_factory();start=clock();begin.record();value=op();end.record();end.synchronize();finish=clock()
  if not stream_is_expected():raise ValueError('CUDA stream changed during measurement')
  ms=float(begin.elapsed_time(end))
  if not math.isfinite(ms) or ms<0 or finish<start:raise ValueError('invalid timing')
  # No reduction/hash/synchronization is part of the recorded interval.
  if not equal(value,references[component]) or fingerprint(value)!=fingerprints[component]:
   journal({'status':'measurement_failed','job':job,'reason':'full_output_mismatch'});raise ValueError('measured full output differs')
  if not hooks_clean():raise ValueError('diagnostic hook remains after operation')
  scope='three-stream selection and visual-mask staging; excludes decoder DS addition' if component=='ds_injection' else 'isolated component diagnostic; not TTFT'
  journal({'status':'measured','job':job,'gpu_event_ms':ms,'host_wall_ms':(finish-start)/1e6,'full_output_equal':True,'output_sha256':fingerprints[component],'scope':scope})
  seen[component]+=1
 if seen!={x:20 for x in COMPONENTS} or warmed!=set(COMPONENTS):raise ValueError('missing or duplicated pilot jobs')
 return {'status':'completed','measurements':60,'warmup_checks':9,'per_component':seen,'full_outputs_checked':69}
