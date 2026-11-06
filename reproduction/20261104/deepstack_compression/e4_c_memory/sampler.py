"""Sampling engine only. No CLI or permission to start a GPU workload.

Callers must supply independently established container/host identity and a
separately bounded supervisor. Persist every event with the supplied callback.
"""
from contract import identity

def parse_sample(output,uuid,pid):
 target=[];others=[]
 for line in output.splitlines():
  if not line.strip():continue
  parts=[p.strip() for p in line.split(',')]
  if len(parts)!=3:raise ValueError('malformed compute process record')
  gpu,raw_pid,raw_mem=parts
  if gpu!=uuid:continue
  if not raw_pid.isdecimal() or not raw_mem.isdecimal():raise ValueError('process memory not numeric')
  item={'pid':int(raw_pid),'used_bytes':int(raw_mem)*1048576}
  (target if item['pid']==pid else others).append(item)
 if len(target)>1:raise ValueError('duplicate target process')
 return (target[0]['used_bytes'] if target else None),others

def sample_process(expected,alive,query,clock,sleep,record,notify_ready,stopped_event,interval_ns,timeout_ns):
 identity(expected)
 if type(interval_ns) is not int or interval_ns<1 or type(timeout_ns) is not int or timeout_ns<=interval_ns:raise ValueError('invalid sampling bounds')
 if not alive(expected):raise ValueError('host process identity not alive')
 begin=clock();started={**expected,'status':'sampling','monotonic_ns':begin};record(started)
 first=None;previous=None;seen=False;count=0
 try:
  while True:
   now=clock()
   if now-begin>timeout_ns:raise TimeoutError('sampling deadline exceeded')
   query_start=clock();used,others=parse_sample(query(),expected['gpu_uuid'],expected['host_pid']);stamp=clock();running=alive(expected)
   if not running and used is not None:raise ValueError('target process identity ended before memory sample')
   if query_start>stamp or stamp-query_start>expected['max_sample_gap_ns'] or (previous is not None and (query_start<previous[1] or stamp-previous[0]>expected['max_sample_gap_ns'])):raise ValueError('sampling gap or clock ordering')
   previous=(query_start,stamp);sample={'type':'sample','query_start_ns':query_start,'query_end_ns':stamp,'monotonic_ns':stamp,'used_bytes':used,'other_gpu_processes':others,**{k:expected[k] for k in ('host_pid','start_ticks','boot_id','gpu_uuid')}}
   record(sample);count+=1;seen|=used is not None
   if others:raise ValueError('other GPU process present')
   if first is None:
    first=sample
    if not running:raise ValueError('worker exited before sampling ready')
    notify_ready(started,first)
   elif not running:
    stopped_ns=stopped_event(expected)
    if stopped_ns is None or type(stopped_ns) is not int or stopped_ns>query_start:
     sleep(interval_ns);continue
    if not seen:raise ValueError('target process memory was never observed')
    result={'type':'summary','status':'completed','samples':count,'target_observed':True,'monotonic_ns':stamp}
    record(result);return result
   sleep(interval_ns)
 except BaseException as exc:
  record({'type':'failure','error_type':type(exc).__name__,'message':str(exc),'monotonic_ns':clock(),'samples':count})
  raise
