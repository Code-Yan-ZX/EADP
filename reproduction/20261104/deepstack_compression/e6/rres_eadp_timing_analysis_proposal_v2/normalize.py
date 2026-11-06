"""Pure adapters for frozen execution schemas, no filesystem access."""

def completed_records(process,journal):
 requests=process['requests']
 if len(journal)!=2*len(requests):raise ValueError('incomplete journal')
 result=[]
 for i,request in enumerate(requests):
  start,done=journal[2*i:2*i+2]
  if (start.get('event')!='started' or start.get('index')!=i or start.get('request')!=request or
      done.get('event')!='completed' or done.get('index')!=i or done.get('request')!=request or done.get('arm')!=process['arm']):
   raise ValueError('journal started/completed sequence')
  result.append(done)
 return result

def observer(records,clock):
 if not isinstance(clock,str) or not clock or not records or records[-1].get('event')!='observer_end' or records[-1].get('reason')!='stop':raise ValueError('observer not stopped')
 end=records[-1];body=records[:-1]
 if len(body)%2 or end.get('samples')!=len(body)//2 or not body:raise ValueError('observer sample inventory')
 namespace=end.get('time_namespace');ps=[];docker=[];sync=[];status=True
 if not isinstance(namespace,str) or not namespace or type(end.get('end_ns')) is not int:raise ValueError('observer clock')
 for i in range(len(body)//2):
  sample,flush=body[2*i:2*i+2]
  if (sample.get('event')!='snapshot' or flush.get('event')!='logging_complete' or sample.get('index')!=i or flush.get('index')!=i or sample.get('time_namespace')!=namespace or flush.get('time_namespace')!=namespace):raise ValueError('observer pairing')
  p=sample['ps'];d=sample['docker'];a,b,c,e,f=p['query_start_ns'],p['query_end_ns'],d['query_start_ns'],d['query_end_ns'],flush['first_fsync_end_ns']
  if any(type(x) is not int for x in (a,b,c,e,f)) or not 0<=a<=b<=c<=e<=f<=end['end_ns'] or (ps and a<sync[-1][1]):raise ValueError('observer times')
  ps.append([a,b]);docker.append([c,e]);sync.append([e,f]);status=status and p.get('returncode')==0 and d.get('returncode')==0
 def group(values):return {'time_namespace':namespace,'clock':clock,'windows':values,'coverage_start_ns':ps[0][0] if status else None,'coverage_end_ns':end['end_ns'] if status else None}
 return {'ps_query':group(ps),'docker_query':group(docker),'first_fsync':group(sync),'queries_all_succeeded':status}

def supervisor_health(records,clock):
 if not records:return None
 # The ordinary supervisor's historical schema has UTC but no monotonic
 # window or time namespace. UTC cannot locate its interference precisely.
 if any('time_namespace' not in x or 'health_snapshot_window_monotonic_ns' not in x for x in records):return None
 namespaces={x.get('time_namespace') for x in records}
 if len(namespaces)!=1 or None in namespaces:raise ValueError('health namespace')
 spans=[]
 for row in records:
  pair=row.get('health_snapshot_window_monotonic_ns')
  if not isinstance(pair,list) or len(pair)!=2 or any(type(x) is not int for x in pair) or pair[0]>pair[1]:raise ValueError('health window')
  spans.append(pair)
 # Generic supervisor terminal files lack precise coverage bounds. A recorded
 # overlap is usable, but a negative observation stays unknown.
 return {'time_namespace':next(iter(namespaces)),'clock':clock,'windows':spans,'coverage_start_ns':None,'coverage_end_ns':None}

def automatic_health_audit(events,metadata):
 # Event records are disclosed without assigning them to request clocks: the
 # event schema itself does not contain a time namespace or clock identity.
 if not isinstance(events,list) or not isinstance(metadata,dict):raise ValueError('automatic health audit schema')
 if any(not isinstance(x,dict) for x in events):raise ValueError('automatic health event schema')
 failures=[x for x in events if x.get('event')=='health_failed' or (x.get('event')=='health_completed' and x.get('returncode')!=0)]
 lifecycle=[x for x in events if x.get('event') in ('monitor_started','monitor_finished','signal','health_failed') or (x.get('event')=='health_completed' and x.get('returncode')!=0)]
 return {'snapshot_metadata':metadata,'event_count':len(events),'lifecycle_and_failure_events':lifecycle,
         'failure_count':len(failures),'coverage':'unknown','failure_overlap':'unknown',
         'reason':'event time namespace and complete monitor coverage are not established'}

def automatic_health(records,clock):
 if not records:return None
 namespaces={x.get('time_namespace') for x in records}
 if len(namespaces)!=1 or None in namespaces:raise ValueError('automatic health namespace')
 spans=[]
 for row in records:
  if row.get('event')!='manual_health_check':raise ValueError('health event')
  start,end=row.get('start_monotonic_ns'),row.get('end_monotonic_ns')
  if type(start) is not int or type(end) is not int or start>end:raise ValueError('automatic health window')
  spans.append([start,end])
 return {'time_namespace':next(iter(namespaces)),'clock':clock,'windows':spans,'coverage_start_ns':None,'coverage_end_ns':None}
