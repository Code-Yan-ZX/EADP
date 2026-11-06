"""CPU-only frozen-cohort design; no GPU launcher or output inspection."""
import hashlib
TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST');BUDGETS=(128,256,512,1024);PROTOCOLS=('natural','fixed32');QUOTAS=(11,11,10);SEED='20261105|e4c-four-k-pilot-v1'
def rank(*parts):return hashlib.sha256('|'.join(map(str,(SEED,*parts))).encode('utf8')).hexdigest()
def build(cohort,identity):
 inventory={};warm=[];measure=[]
 for task,quota in zip(TASKS,QUOTAS):
  c=cohort[task];m=c['measured'];w=c['warmup']
  if len(m)!=64 or len(w)!=10 or len(set(m))!=64 or len(set(w))!=10 or set(m)&set(w):raise ValueError('invalid 64/10 cohort')
  selected=sorted(m,key=lambda g:rank('pilot',task,g))[:quota];inventory[task]={'measured':selected,'warmup':list(w)}
  for phase,groups in (('measure',selected),('warmup',w)):
   for group in groups:
    row=identity[(task,group)]
    if row['task']!=task or row['group_id']!=group or not str(row['qid']) or not row['image_path'] or len(row['image_sha256'])!=64 or len(row['message_sha256'])!=64:raise ValueError('input identity invalid')
    (measure if phase=='measure' else warm).append(row)
 jobs=[]
 for phase,items,repeats in (('warmup',warm,1),('measure',measure,5)):
  for rep in range(repeats):
   for row in sorted(items,key=lambda r:rank('image',phase,rep,r['task'],r['group_id'])):
    arms=[(k,p) for k in BUDGETS for p in PROTOCOLS]
    for k,p in sorted(arms,key=lambda z:rank('arm',phase,rep,row['group_id'],z[0],z[1])):
     jobs.append({'phase':phase,'replicate':rep,'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'budget':k,'protocol':p,'order':len(jobs)})
 if len(jobs)!=1520:raise ValueError('bad schedule denominator')
 return inventory,jobs
