"""Pure six-process identity schedule; no container or GPU execution."""
ARMS=('author','wavefront')

def schedule(images):
 if len(images)!=3: raise ValueError('three anchors required')
 anchors=[x['anchor'] for x in images]
 keys=[(x['task'],str(x['qid'])) for x in anchors]
 if len(set(keys))!=3 or len({x['task'] for x in anchors})!=3: raise ValueError('anchor duplication')
 jobs=[]
 for index,anchor in enumerate(anchors):
  # Explicitly alternate order; these are identity tests, not latency tests.
  for arm in (ARMS if index%2==0 else ARMS[::-1]):
   jobs.append({'order':len(jobs),'task':anchor['task'],'qid':str(anchor['qid']),
                'group_id':anchor['group_id'],'image_sha256':anchor['image_sha256'],
                'message_sha256':anchor['message_sha256'],'arm':arm})
 return jobs

def validate_history(jobs, events):
 """Reject missing/duplicate/out-of-order outcomes; no failed arm can pass."""
 if len(jobs)!=6 or len(events)!=12:raise ValueError('incomplete identity schedule')
 containers=set()
 for n,job in enumerate(jobs):
  start,end=events[2*n:2*n+2]
  if start.get('event')!='started' or end.get('event')!='completed' or start.get('job')!=job or end.get('job')!=job:
   raise ValueError('identity events out of order')
  cid=start.get('container_id')
  if not isinstance(cid,str) or len(cid)!=64 or any(c not in '0123456789abcdef' for c in cid) or cid in containers:
   raise ValueError('missing or reused container identity')
  containers.add(cid)
  if (end.get('container_id')!=cid or end.get('returncode')!=0 or end.get('container_exit_code')!=0
      or end.get('container_stopped_verified') is not True or end.get('timed_out') is not False
      or end.get('interrupted') is not None or end.get('result_status')!='passed'):
   raise ValueError('failed identity arm')
 return {'images':3,'jobs':6,'isolated_containers':6}
