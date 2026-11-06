"""Pre-specified three-anchor natural TTFT pilot; no GPU admission."""
import hashlib
BUDGETS=(128,256,512,1024);SEED='eadp-budget-ttft-20261105-001';REPETITIONS=3
def order(*parts):return hashlib.sha256('|'.join(map(str,(SEED,*parts))).encode()).hexdigest()
def schedule(anchors):
 if len(anchors)!=3 or len({r['task'] for r in anchors})!=3 or len({r['group_id'] for r in anchors})!=3 or any(r.get('split')!='DEV' for r in anchors):raise ValueError('exact three DEV image groups required')
 rows=[]
 # All warmup samples precede all measured samples; each block is complete.
 for phase,reps in (('warmup',1),('measure',REPETITIONS)):
  for rep in range(reps):
   for r in sorted(anchors,key=lambda x:order('anchor',phase,rep,x['task'],x['group_id'])):
    for k in sorted(BUDGETS,key=lambda x:order('budget',phase,rep,r['group_id'],x)):
     rows.append({'task':r['task'],'qid':str(r['qid']),'group_id':r['group_id'],'phase':phase,'replicate':rep,'budget':k,'order':len(rows)})
 return rows
def validate(jobs,anchors):
 expected=schedule(anchors)
 if jobs!=expected or len(jobs)!=48:raise ValueError('seeded complete 48-job schedule mismatch')
 return True
