"""Pure accepted-data analysis adapter; caller must gate all inputs first."""
import metrics,normalize,windows

def analyze(schedule,journals,process_clocks,observer_records,observer_clock,run_health,observer_health,automatic_health,run_health_clock=None,observer_health_clock=None,automatic_health_clock=None,automatic_events=None,automatic_metadata=None):
 if len(journals)!=12 or len(process_clocks)!=12:raise ValueError('12 process journals and clock records required')
 completed=[normalize.completed_records(p,j) for p,j in zip(schedule['processes'],journals)]
 result=metrics.summarize(schedule,completed)
 obs=normalize.observer(observer_records,observer_clock)
 rh=normalize.supervisor_health(run_health,run_health_clock)
 oh=normalize.supervisor_health(observer_health,observer_health_clock)
 mh=normalize.automatic_health(automatic_health,automatic_health_clock)
 result['automatic_health_audit']=normalize.automatic_health_audit(automatic_events or [],automatic_metadata or {})
 for row in result['per_job']:
  c=process_clocks[row['process']];ns=c.get('time_namespace');mono=c.get('monotonic_clock');perf=c.get('perf_counter_clock')
  if not isinstance(ns,str) or not ns:raise ValueError('request time namespace missing')
  # Never infer perf-counter-to-monotonic equivalence from UTC or system name.
  clock=mono if isinstance(mono,str) and mono and mono==perf else 'unverified request clock'
  groups={'observer_query':obs['ps_query'],'observer_first_fsync':obs['first_fsync'],'run_supervisor_health':rh,'observer_supervisor_health':oh,'manual_check':mh}
  t=row['times'];flags=windows.annotate(t['t0_ns'],t['response_end_ns'],ns,clock,groups)
  groups['observer_query']=obs['docker_query'];other=windows.annotate(t['t0_ns'],t['response_end_ns'],ns,clock,groups)
  flags['observer_ps_query']=flags.pop('observer_query');flags['observer_docker_query']=other['observer_query'];row['windows']=flags
  row['time_namespace']=ns;row['clock_implementation']=clock
 result['limitations']=['Each group is one image; five repeats are reduced within image.',
   'Observer and health findings describe recorded windows only; no recorded overlap is not evidence of no interference.',
   'Unknown coverage/clock, second fsync, automatic-health log fsync and unobserved host activity remain unknown.',
   'Posthoc small same-DEV pilot; no equal-compute, causal or confirmatory inference.']
 return result
