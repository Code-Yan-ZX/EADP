"""Blinded DEV latency analysis primitives; callers must establish all gates."""
import math
from collections import Counter
TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST');MODES=('C-ON','C-DROP','C-SKIP');PROTOCOLS=('natural','fixed32')
CONTRASTS=(('C-DROP','C-ON'),('C-SKIP','C-ON'),('C-SKIP','C-DROP'))
def validate_windows(official,manual,namespace):
 if not official or not manual:raise ValueError('monitor windows missing')
 def pair(v):
  if not isinstance(v,list) or len(v)!=2 or any(type(x) is not int or x<0 for x in v) or v[0]>v[1]:raise ValueError('invalid monitor interval')
  return v
 report=[]
 for i,x in enumerate(official):
  if x.get('time_namespace')!=namespace:raise ValueError('official monitor clock')
  report.append({'kind':'supervisor','index':i,'window_ns':pair(x.get('health_snapshot_window_monotonic_ns')),'time_namespace':namespace,'event':x.get('event')})
 for i,x in enumerate(manual):
  if x.get('time_namespace')!=namespace or 'nvml_window_monotonic_ns' not in x:raise ValueError('manual monitor clock/status')
  health=pair(x.get('manual_health_window_monotonic_ns'));nvml=x['nvml_window_monotonic_ns']
  report.append({'kind':'manual_health','index':i,'window_ns':health,'time_namespace':namespace,'timestamp':x.get('timestamp'),'check':x.get('check')})
  if nvml is None:
   if 'no additional NVML query' not in x.get('gpu_check',''):raise ValueError('unknown NVML status')
  else:
   nvml=pair(nvml)
   if not health[0]<=nvml[0]<=nvml[1]<=health[1]:raise ValueError('NVML outside manual health interval')
   report.append({'kind':'manual_nvml','index':i,'window_ns':nvml,'time_namespace':namespace,'timestamp':x.get('timestamp'),'check':x.get('check')})
 return report

def summarize_windows(records,windows):
 out=[]
 for window in windows:
  field={'supervisor':'monitor_snapshot_indices','manual_health':'manual_health_indices','manual_nvml':'manual_nvml_indices'}[window['kind']];index=window['index'];counts=[]
  for phase in ('warmup','measure'):
   for task in TASKS:
    for mode in MODES:
     for protocol in PROTOCOLS:
      values=[r for r in records if r['phase']==phase and r['task']==task and r['mode']==mode and r['protocol']==protocol]
      counts.append({'phase':phase,'task':task,'mode':mode,'protocol':protocol,'n_total':len(values),'n_overlap':sum(index in r[field] for r in values)})
  out.append({**window,'n_overlap':sum(index in r[field] for r in records),'counts':counts})
 return out

def build_records(schedule,inputs,events,formal,manual,namespace,windows):
 jobs=schedule['schedule'];bykey={};groups={}
 for r in inputs:
  key=(r['task'],str(r['qid']));g=(r['task'],r['group_id'])
  if key in bykey or g in groups or r.get('split') not in (None,'DEV'):raise ValueError('duplicate/non-DEV accepted input')
  if any(not isinstance(r.get(k),str) or len(r[k])!=64 for k in ('image_sha256','message_sha256')):raise ValueError('input digest missing')
  bykey[key]=r;groups[g]=r
 if len(bykey)!=384 or len(groups)!=384 or len(jobs)!=5940 or len(events)!=11880 or len(formal)!=11880 or len(manual)!=5940:raise ValueError('input/event inventory')
 out=[];seen=set();measured={};counts=Counter()
 for idx,(job,raw,done,ann,ann_done,man) in enumerate(zip(jobs,events[::2],events[1::2],formal[::2],formal[1::2],manual)):
  if job.get('order')!=idx or raw.get('status')!='generated' or done.get('status')!='completed' or ann.get('status')!='generated' or ann_done.get('status')!='completed' or any(x.get('job')!=job for x in (raw,done,ann,ann_done,man)):raise ValueError('job/order mismatch')
  task=job.get('task');mode=job.get('mode');protocol=job.get('protocol');phase=job.get('phase');rep=job.get('replicate');key=(task,str(job.get('qid')));g=(task,job.get('group_id'))
  if task not in TASKS or mode not in MODES or protocol not in PROTOCOLS or phase not in ('warmup','measure') or key not in bykey or g not in groups or groups[g] is not bykey[key]:raise ValueError('task/question/image identity mismatch')
  if type(rep) is not int or (phase=='measure' and rep not in range(5)) or (phase=='warmup' and rep!=0):raise ValueError('replicate mismatch')
  identity=(phase,task,job['group_id'],mode,protocol,rep)
  if identity in seen:raise ValueError('duplicate job')
  seen.add(identity);counts[phase]+=1
  if phase=='measure':
   ref=bykey[key];f=(task,job['group_id']);ident=(str(job['qid']),ref['image_sha256'],ref['message_sha256'])
   if f in measured and measured[f]!=ident:raise ValueError('same image group changed question or input')
   measured[f]=ident
  value=raw['result'];b=value['boundaries'];t0=b.get('t0_before_image_read_ns');first=b.get('first_token_selected_ns');end=b.get('generation_complete_ns')
  if (value.get('mode')!=mode or value.get('protocol')!=protocol or b.get('time_namespace')!=namespace or b.get('clock_implementation')!='clock_gettime(CLOCK_MONOTONIC)' or
      any(type(x) is not int or x<0 for x in (t0,first,end)) or not t0<first<=end or b.get('first_criteria_length')!=1 or b.get('explicit_sync_boundaries')!=['before_t0','first_token','generation_end']):raise ValueError('invalid or incompatible timing boundaries')
  if any(type(x.get(k)) is not bool for x,k in ((ann,'monitor_overlap'),(man,'manual_health_overlap'),(man,'manual_nvml_overlap'))):raise ValueError('unknown monitoring overlap')
  expected_indices={kind:[w['index'] for w in windows if w['kind']==kind and w['window_ns'][0]<=end and t0<=w['window_ns'][1]] for kind in ('supervisor','manual_health','manual_nvml')}
  if (ann.get('monitor_snapshot_indices')!=expected_indices['supervisor'] or ann['monitor_overlap']!=bool(expected_indices['supervisor']) or
      man.get('manual_health_indices')!=expected_indices['manual_health'] or man['manual_health_overlap']!=bool(expected_indices['manual_health']) or
      man.get('manual_nvml_indices')!=expected_indices['manual_nvml'] or man['manual_nvml_overlap']!=bool(expected_indices['manual_nvml'])):raise ValueError('monitor window annotation mismatch')
  n=value.get('output_tokens');eos=value.get('eos');truncated=value.get('truncated')
  if type(n) is not int or n<1 or n>2048 or (protocol=='fixed32' and n!=32) or type(eos) is not bool or type(truncated) is not bool:raise ValueError('output metadata')
  out.append({'task':task,'group_id':job['group_id'],'qid':str(job['qid']),'mode':mode,'protocol':protocol,'phase':phase,'replicate':rep,
              'ttft_ms':(first-t0)/1e6,'end_to_end_ms':(end-t0)/1e6,'output_tokens':n,'eos':eos,'truncated':truncated,
              'monitor_overlap':ann['monitor_overlap'],'monitor_snapshot_indices':ann['monitor_snapshot_indices'],'manual_health_overlap':man['manual_health_overlap'],'manual_health_indices':man['manual_health_indices'],'manual_nvml_overlap':man['manual_nvml_overlap'],'manual_nvml_indices':man['manual_nvml_indices']})
 if counts!={'measure':5760,'warmup':180}:raise ValueError('phase counts')
 for task in TASKS:
  gs={g for t,g in measured if t==task}
  if len(gs)!=64:raise ValueError('measured group count')
  for g in gs:
   for mode in MODES:
    for protocol in PROTOCOLS:
     if any(('measure',task,g,mode,protocol,rep) not in seen for rep in range(5)):raise ValueError('incomplete paired group')
 return out

def analyze_task(rows,draws):
 import numpy as np
 if not rows:raise ValueError('empty task')
 groups=sorted({r['group_id'] for r in rows});n=len(groups);draws=np.asarray(draws)
 if draws.ndim!=2 or draws.shape[1]!=n or np.any(draws<0) or np.any(draws>=n):raise ValueError('bootstrap draw shape')
 arrays={};lookup={}
 for r in rows:
  k=(r['group_id'],r['mode'],r['protocol'],r['replicate'])
  if r['phase']!='measure' or r['mode'] not in MODES or r['protocol'] not in PROTOCOLS or type(r['replicate']) is not int or r['replicate'] not in range(5) or k in lookup:raise ValueError('invalid measured inventory')
  if any(not math.isfinite(r[m]) or r[m]<=0 for m in ('ttft_ms','end_to_end_ms')):raise ValueError('nonpositive or nonfinite timing')
  lookup[k]=r
 if len(lookup)!=n*len(MODES)*len(PROTOCOLS)*5:raise ValueError('incomplete measured matrix')
 report={'n_image_groups':n,'repeats_per_group':5,'measurements_per_mode_protocol':n*5,'protocols':{}}
 for protocol in PROTOCOLS:
  per={'modes':{},'metrics':{}}
  for mode in MODES:
   selected=[lookup[(g,mode,protocol,rep)] for g in groups for rep in range(5)]
   tok=np.asarray([r['output_tokens'] for r in selected]);per['modes'][mode]={'n':len(selected),'output_tokens':{'min':int(tok.min()),'median':float(np.quantile(tok,.5,method='linear')),'p95':float(np.quantile(tok,.95,method='linear')),'max':int(tok.max())},'eos':sum(r['eos'] for r in selected),'truncated':sum(r['truncated'] for r in selected)}
  for metric in ('ttft_ms','end_to_end_ms'):
   for quantile,label in ((.5,'median'),(.95,'p95')):
    points={};boot={}
    for mode in MODES:
     a=np.asarray([[lookup[(g,mode,protocol,rep)][metric] for rep in range(5)] for g in groups],dtype=float)
     points[mode]=float(np.quantile(a.reshape(-1),quantile,method='linear'))
     boot[mode]=np.quantile(a[draws].reshape(len(draws),-1),quantile,axis=1,method='linear')
    comparisons={}
    for numerator,denominator in CONTRASTS:
     if points[denominator]<=0 or np.any(boot[denominator]<=0):raise ValueError('invalid ratio denominator')
     diff=boot[numerator]-boot[denominator];ratio=boot[numerator]/boot[denominator]
     bounds=lambda x:[float(v) for v in np.quantile(x,[.025,.975],method='linear')]
     comparisons[numerator+'_minus_'+denominator]={'difference_ms':points[numerator]-points[denominator],'difference_95_ci_ms':bounds(diff),'ratio':points[numerator]/points[denominator],'ratio_95_ci':bounds(ratio),'relative_change_pct':100*(points[numerator]/points[denominator]-1),'relative_change_95_ci_pct':bounds(100*(ratio-1))}
    per['metrics'][metric+'_'+label]={'mode_quantiles_ms':points,'paired_contrasts':comparisons}
  report['protocols'][protocol]=per
 return report

def analyze(records,replicates=10000,seed=20261104):
 import numpy as np
 if type(replicates) is not int or replicates<1:raise ValueError('bootstrap count')
 rng=np.random.default_rng(seed);tasks={}
 for task in TASKS:
  rows=[r for r in records if r['task']==task and r['phase']=='measure'];groups=sorted({r['group_id'] for r in rows})
  if len(groups)!=64:raise ValueError('expected 64 measured groups per task')
  draws=rng.integers(0,64,size=(replicates,64));tasks[task]=analyze_task(rows,draws)
 return tasks
