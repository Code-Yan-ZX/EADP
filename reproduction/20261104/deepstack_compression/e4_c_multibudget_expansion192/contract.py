"""Pure frozen 192-image selection and deterministic proposal; no I/O or GPU."""
import hashlib
from collections import Counter
TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST');KS=(128,256,512,1024);PROTOCOLS=('natural','fixed32')
SELECT='EADP-E4C-192-v1';SCHEDULE='EADP-E4C-192-SCHEDULE-v1'
def digest(*parts):return hashlib.sha256('|'.join(map(str,parts)).encode('utf8')).hexdigest()
def select_rank(row):return (digest(SELECT,row['task'],row['group_id'],str(row['qid'])),row['group_id'],str(row['qid']))
def build(rows,pilot):
 if len(rows)!=384 or Counter(r['task'] for r in rows)!=Counter({t:128 for t in TASKS}):raise ValueError('must contain 384 DEV identities')
 bygroup={(r['task'],r['group_id']):r for r in rows};byquestion={(r['task'],str(r['qid'])):r for r in rows}
 if len(bygroup)!=384 or len(byquestion)!=384 or any(r.get('split')!='DEV' or not isinstance(r.get('group_id'),str) or not r['group_id'] or not str(r.get('qid','')) for r in rows):raise ValueError('duplicate or non-DEV identity')
 old=pilot['input_identities'];excluded={(r['task'],r['group_id']) for r in old}
 if len(old)!=62 or len(excluded)!=62 or not excluded<=set(bygroup):raise ValueError('pilot exclusion inventory')
 for r in old:
  other=bygroup[(r['task'],r['group_id'])]
  if any(str(r.get(k))!=str(other.get(k)) for k in ('task','qid','group_id','image_sha256','message_sha256','image_path')):raise ValueError('pilot identity mismatch')
 measured=[];warm=[];inventory={}
 for task in TASKS:
  elig=[r for r in rows if r['task']==task and (task,r['group_id']) not in excluded]
  if len(elig)<64:raise ValueError('insufficient eligible groups')
  selected=sorted(elig,key=select_rank)[:64];w=pilot['cohort'][task]['warmup']
  if len(w)!=10 or len(set(w))!=10 or any((task,g) not in excluded for g in w):raise ValueError('pilot warmup inventory')
  measured.extend(selected);warm.extend(bygroup[(task,g)] for g in w)
  inventory[task]={'eligible_groups':len(elig),'measured':[str(r['group_id']) for r in selected],'warmup':list(w)}
 if len(measured)!=192 or len(warm)!=30 or {(r['task'],r['group_id']) for r in measured}&excluded:raise ValueError('measurement exclusion')
 jobs=[]
 for phase,items,reps in (('warmup',warm,1),('measure',measured,5)):
  for rep in range(reps):
   ranked=sorted(items,key=lambda r:(digest(SCHEDULE,'image',phase,rep,r['task'],r['group_id'],str(r['qid'])),r['task'],r['group_id'],str(r['qid'])))
   for row in ranked:
    arms=sorted(((k,p) for k in KS for p in PROTOCOLS),key=lambda z:(digest(SCHEDULE,'arm',phase,rep,row['task'],row['group_id'],str(row['qid']),z[0],z[1]),z[0],z[1]))
    for k,p in arms:jobs.append({'phase':phase,'replicate':rep,'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'budget':k,'protocol':p,'order':len(jobs)})
 if len(jobs)!=7920:raise ValueError('job count')
 return {'cohort':inventory,'measurement_identities':measured,'warmup_identities':warm,'jobs':jobs,'GT_opened':False,'predictions_opened':False,'GPU_authorized':False}
