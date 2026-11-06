"""Pure 7920-job environment annotations; no I/O, source authority or GPU."""
from collections import Counter
def overlap(windows,a,b):
 if type(a) is not int or type(b) is not int or a>=b:raise ValueError('invalid job interval')
 return [i for i,(x,y) in enumerate(windows) if x<=b and a<=y]
def validate_windows(values):
 if not isinstance(values,list):raise ValueError('window list missing')
 for v in values:
  if not isinstance(v,list) or len(v)!=2 or any(type(x) is not int or x<0 for x in v) or v[0]>v[1]:raise ValueError('malformed bounded window')
def report(jobs,events,window_sets,namespace):
 if not isinstance(namespace,str) or not namespace or len(jobs)!=7920 or len(events)!=23760:raise ValueError('incomplete inputs')
 keys=('model_health','query','query_first_fsync','watcher_health','manual_health')
 if set(window_sets)!=set(keys):raise ValueError('window inventory')
 for k in keys:validate_windows(window_sets[k])
 if not window_sets['model_health'] or not window_sets['query'] or not window_sets['watcher_health'] or len(window_sets['query'])!=len(window_sets['query_first_fsync']):raise ValueError('missing observer evidence')
 for a,b in zip(window_sets['query'],window_sets['query_first_fsync']):
  if a[0]!=b[0] or a[1]>b[1]:raise ValueError('query/fsync mismatch')
 details=[];counts={};phases=Counter()
 for i,j in enumerate(jobs):
  t=events[3*i:3*i+3]
  if j.get('order')!=i or [x.get('event') for x in t]!=['job_started','raw_generation','job_completed'] or any(x.get('job')!=j for x in t):raise ValueError('job order')
  r=t[2]['record'];raw=t[1]['raw']
  for field in ('task','qid','group_id','budget','protocol'):
   if raw.get(field)!=j.get(field) or r.get(field)!=j.get(field):raise ValueError('job identity '+field)
  for field in ('output_ids','eos_ids','generation_scalars','min_new_tokens'):
   if raw.get(field)!=r.get(field):raise ValueError('raw output mismatch')
  v=r['boundaries'];a=v['t0_before_image_read_ns'];b=v['first_token_selected_ns'];c=v['generation_complete_ns']
  if v.get('time_namespace')!=namespace or type(a) is not int or type(b) is not int or type(c) is not int or not 0<=a<b<=c or raw.get('generation_complete_ns')!=c:raise ValueError('time/clock mismatch')
  hits={k:{'ttft':overlap(window_sets[k],a,b),'generation':overlap(window_sets[k],a,c)} for k in keys}
  row={k:j[k] for k in ('order','phase','task','qid','group_id','budget','protocol','replicate')};row.update(t0_before_image_read_ns=a,first_token_selected_ns=b,generation_complete_ns=c,token_count=len(r['output_ids']),eos=r['eos'],truncated=r['truncated'],overlap_indices=hits);details.append(row);phases[j['phase']]+=1
  if j['phase']=='measure':
   key='/'.join((j['task'],j['protocol'],str(j['budget'])));value=counts.setdefault(key,{'jobs':0,'overlaps':{k:{'ttft':0,'generation':0} for k in keys}});value['jobs']+=1
   for k in keys:
    for kind in ('ttft','generation'):value['overlaps'][k][kind]+=bool(hits[k][kind])
 if phases!=Counter({'warmup':240,'measure':7680}):raise ValueError('phase denominator')
 if len(counts)!=24 or any(v['jobs']!=320 for v in counts.values()):raise ValueError('task/budget/protocol denominator')
 start=min(x['t0_before_image_read_ns'] for x in details);end=max(x['generation_complete_ns'] for x in details);period=3600*10**9;slack=300*10**9;combined=window_sets['model_health']+window_sets['watcher_health'];hours=[];n=1
 while start+n*period<=end:
  target=start+n*period;m=[i for i,(x,y) in enumerate(window_sets['manual_health']) if x<=target+slack and y>=target-slack];h=[i for i,(x,y) in enumerate(combined) if x<=target+slack and y>=target-slack]
  hours.append({'hour':n,'manual_indices':m,'supervisor_health_indices':h,'manual_status':'observed' if m else 'unknown','supervisor_health_status':'observed' if h else 'unknown'});n+=1
 return {'jobs_detail':details,'health_overlap':counts,'hourly_health_coverage':hours,'manual_check_evidence':'recorded' if window_sets['manual_health'] else 'unknown','unknown':'observer second fsync, manual log fsync, marker write, unrecorded residual interference','samples_excluded':0}
