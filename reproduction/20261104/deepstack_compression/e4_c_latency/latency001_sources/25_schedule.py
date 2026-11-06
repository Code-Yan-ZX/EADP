"""Deterministic warmup/measurement jobs from the accepted E4-C schedule."""
from contract import TASKS,make_schedule,validate_schedule

def build(rows,excluded,stage):
 full=make_schedule(rows,excluded);validate_schedule(full)
 if stage=='full':return full
 if stage!='smoke':raise ValueError('unknown timing stage')
 selected={task:{'measure':set(full['cohort'][task]['measured'][:2]),'warmup':set(full['cohort'][task]['warmup'][:2])} for task in TASKS}
 jobs=[]
 for source in full['schedule']:
  if source['group_id'] not in selected[source['task']][source['phase']]:continue
  if source['phase']=='measure' and source['replicate']!=0:continue
  job=dict(source,source_order=source['order'],order=len(jobs));jobs.append(job)
 if len(jobs)!=72:raise ValueError('timing smoke inventory')
 return {'seed':full['seed'],'stage':'smoke','scope':'DEV batch1 warm-cache diagnostic; no GT or scores',
         'cohort':{task:{k:sorted(v) for k,v in selected[task].items()} for task in TASKS},'schedule':jobs,
         'counts':{'warmup':36,'measured':36},'derivation':'first two frozen measured and warmup groups per task; full schedule relative order preserved'}
