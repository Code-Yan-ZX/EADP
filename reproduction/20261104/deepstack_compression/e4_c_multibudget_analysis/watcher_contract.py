"""Validate background observation windows without deleting any sample."""
from datetime import datetime

def utc(s):
 if not isinstance(s,str) or not s:raise ValueError('missing UTC time')
 v=datetime.fromisoformat(s.replace('Z','+00:00'))
 if v.tzinfo is None:raise ValueError('timezone missing')
 return v

def validate(samples,health,manual,namespace,model_start_utc,model_end_utc):
 if not isinstance(namespace,str) or not namespace or len(samples)<3 or len(samples)%2!=1:raise ValueError('watcher source incomplete')
 end=samples[-1]
 if end.get('event')!='watch_end' or end.get('time_namespace')!=namespace or end.get('reason')!='stop_marker' or end.get('samples')!=(len(samples)-1)//2 or type(end.get('end_ns')) is not int:raise ValueError('watcher did not stop cleanly')
 query=[];query_fsync=[];prev=-1
 for i in range((len(samples)-1)//2):
  a,b=samples[2*i:2*i+2];x=a.get('query_start_ns');y=a.get('query_end_ns');z=b.get('first_fsync_end_ns')
  if (a.get('event')!='snapshot' or b.get('event')!='logging_complete' or a.get('index')!=i or b.get('index')!=i or a.get('time_namespace')!=namespace or b.get('time_namespace')!=namespace or not all(type(v) is int for v in (x,y,z)) or not prev<x<=y<=z or b.get('query_start_ns')!=x or a.get('returncode')!=0 or not isinstance(a.get('stdout'),str)):raise ValueError('invalid or failed watcher snapshot')
  for s in (a.get('query_start_utc'),a.get('query_end_utc'),b.get('first_fsync_end_utc')):utc(s)
  query.append([x,y]);query_fsync.append([x,z]);prev=z
 if end['end_ns']<prev or utc(samples[0]['query_start_utc'])>utc(model_start_utc) or utc(end.get('end_utc'))<utc(model_end_utc):raise ValueError('watcher does not cover actual run')
 if not isinstance(health,list) or not health:raise ValueError('watcher health unknown')
 health_windows=[]
 for row in health:
  window=row.get('health_snapshot_window_monotonic_ns')
  if row.get('time_namespace')!=namespace or not isinstance(window,list) or len(window)!=2 or any(type(v) is not int for v in window) or window[0]>window[1]:raise ValueError('invalid watcher health window')
  health_windows.append(window)
 if not isinstance(manual,list):raise ValueError('manual health evidence invalid')
 manual_windows=[];prev=-1
 for row in manual:
  x=row.get('start_monotonic_ns');y=row.get('end_monotonic_ns')
  if row.get('event')!='manual_health_check' or row.get('time_namespace')!=namespace or type(x) is not int or type(y) is not int or x<prev or x>y or not isinstance(row.get('details'),dict):raise ValueError('manual health window invalid')
  utc(row.get('start_utc'));utc(row.get('end_utc'));manual_windows.append([x,y]);prev=y
 return {'query':query,'query_first_fsync':query_fsync,'watcher_health':health_windows,'manual_health':manual_windows,'watch_end_ns':end['end_ns'],'unknown':'second log fsync, manual log fsync, stop marker write and residual unobserved activity'}
def overlap(windows,start,end):
 if type(start) is not int or type(end) is not int or start>=end:raise ValueError('invalid job window')
 return [i for i,(a,b) in enumerate(windows) if a<=end and start<=b]
def hourly_coverage(start,end,manual,health):
 if type(start) is not int or type(end) is not int or start>end:raise ValueError('hourly window')
 span=3600*10**9;slack=300*10**9;out=[];n=1
 while start+n*span<=end:
  target=start+n*span
  matches=[i for i,(a,b) in enumerate(manual) if a<=target+slack and b>=target-slack]
  observed=[i for i,(a,b) in enumerate(health) if a<=target+slack and b>=target-slack]
  out.append({'hour':n,'manual_indices':matches,'supervisor_health_indices':observed,'manual_status':'observed' if matches else 'unknown','supervisor_health_status':'observed' if observed else 'unknown','status':'some_health_observed' if matches or observed else 'unknown'})
  n+=1
 return out
