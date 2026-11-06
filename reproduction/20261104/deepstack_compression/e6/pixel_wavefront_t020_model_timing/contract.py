"""CPU-only timing and NVML contracts; no model, Docker or GPU entry."""
import math,random
ARMS=('author','wavefront');SEED=20261105;REPETITIONS=3

def schedule(anchors):
 if len(anchors)!=3 or len({x['task'] for x in anchors})!=3 or len({x['group_id'] for x in anchors})!=3:raise ValueError('three distinct anchors required')
 jobs=[]
 for phase,count in (('warmup',1),('measurement',REPETITIONS)):
  for rep in range(count):
   block=[{'task':x['task'],'qid':str(x['qid']),'group_id':x['group_id'],'image_sha256':x['image_sha256'],'message_sha256':x['message_sha256'],'arm':arm,'phase':phase,'repeat':rep} for x in anchors for arm in ARMS]
   random.Random(SEED+(0 if phase=='warmup' else 1)+rep).shuffle(block)
   for job in block:job['order']=len(jobs);jobs.append(job)
 return jobs

def validate_time(job,record,reference):
 if any(record.get(k)!=v for k,v in job.items()):raise ValueError('wrong timed job')
 for key in ('processor','selected_indices','output_ids','eos_ids','generation_scalars'):
  if record.get(key)!=reference.get(key):raise ValueError('same-arm identity '+key)
 start,first,completion=(record.get(k) for k in ('start_monotonic_ns','first_token_monotonic_ns','completion_monotonic_ns'))
 if any(type(x) is not int for x in (start,first,completion)) or not 0<start<first<=completion:raise ValueError('invalid synchronized timing')
 if record.get('first_token_synchronized') is not True or record.get('gpu_synchronized_before_start') is not True or record.get('all_samples_retained') is not True:raise ValueError('timing boundary disclosure')
 if type(record.get('start_unix_ns')) is not int or type(record.get('completion_unix_ns')) is not int or record['start_unix_ns']>record['completion_unix_ns']:raise ValueError('wall clock missing')
 return {'ttft_ns':first-start,'completion_ns':completion-start,'selected_count':len(record['selected_indices'])}

def validate_nvml(samples,pid,uuid,window,max_gap_ns):
 if not samples or type(pid) is not int or pid<=0 or not uuid or type(max_gap_ns) is not int or max_gap_ns<=0:raise ValueError('sampling identity')
 start,end=window
 if type(start) is not int or type(end) is not int or not 0<start<end:raise ValueError('request window')
 prev=None;rows=[]
 for s in samples:
  a,b=s.get('query_start_ns'),s.get('query_end_ns')
  if type(a) is not int or type(b) is not int or a<=0 or b<a or b-a>max_gap_ns or (prev is not None and (a<prev[1] or b-prev[0]>max_gap_ns)):raise ValueError('sampling gap or clock')
  if s.get('gpu_uuid')!=uuid:raise ValueError('wrong device')
  processes=s.get('processes');device=s.get('device')
  if not isinstance(processes,list) or not isinstance(device,dict):raise ValueError('missing process or device inventory')
  targets=[p for p in processes if p.get('pid')==pid and p.get('type')=='C']
  if len(targets)>1:raise ValueError('duplicate compute PID')
  used=targets[0].get('used_mib') if targets else None
  if used is not None and (type(used) is not int or used<0):raise ValueError('invalid target usage')
  for key in ('total_mib','used_mib','free_mib','reserved_mib'):
   value=device.get(key)
   if value is not None and (type(value) is not int or value<0):raise ValueError('invalid device usage')
  rows.append({'intersects_request':a<=end and b>=start,'target_present':bool(targets),'target_mib':used});prev=(a,b)
 if samples[0]['query_start_ns']>start or samples[-1]['query_end_ns']<end:raise ValueError('window not bracketed')
 return rows
