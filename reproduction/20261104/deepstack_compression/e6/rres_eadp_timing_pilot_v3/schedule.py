"""Fixed 12-process, 60-warmup and 320-measurement schedule."""
import hashlib,json
from collections import Counter
TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST');ARMS=('eadp','rres');SEED='20261106-rres-eadp-v1'
def ident(x):return (x['task'],str(x['qid']),x['group_id'])
def rank(task,block,rep,row):return hashlib.sha256(json.dumps([SEED,task,block,rep,*ident(row)],separators=(',',':')).encode()).hexdigest()
def build(anchors,warmups):
 if len(anchors)!=32 or len(warmups)!=30 or Counter(x['task'] for x in anchors)!=dict(zip(TASKS,(11,11,10))) or Counter(x['task'] for x in warmups)!={x:10 for x in TASKS}:raise ValueError('cohort denominator')
 allrows=anchors+warmups
 if len({ident(x) for x in allrows})!=62 or len({(x['task'],str(x['qid'])) for x in allrows})!=62 or len({x['group_id'] for x in allrows})!=62 or any(not isinstance(x.get('group_id'),str) or not x['group_id'] or x.get('qid') is None or isinstance(x['qid'],bool) or not str(x['qid']) for x in allrows):raise ValueError('cohort overlap')
 jobs=[]
 for ti,task in enumerate(TASKS):
  measured=[x for x in anchors if x['task']==task];warm=[x for x in warmups if x['task']==task]
  for block in range(2):
   selected=measured[block::2];warmblock=warm[block*5:(block+1)*5];order=ARMS if (ti*2+block)%2==0 else ARMS[::-1]
   for arm in order:
    requests=[]
    for x in warmblock:requests.append({'phase':'warmup','replicate':0,'task':task,'qid':str(x['qid']),'group_id':x['group_id']})
    for rep in range(5):
     for x in sorted(selected,key=lambda r:rank(task,block,rep,r)):requests.append({'phase':'measured','replicate':rep,'task':task,'qid':str(x['qid']),'group_id':x['group_id']})
    jobs.append({'order':len(jobs),'task':task,'block':block,'arm':arm,'requests':requests})
 return jobs
def validate(jobs,anchors,warmups):
 if jobs!=build(anchors,warmups):raise ValueError('schedule differs from fixed construction')
 if len(jobs)!=12 or sum(x['phase']=='warmup' for j in jobs for x in j['requests'])!=60 or sum(x['phase']=='measured' for j in jobs for x in j['requests'])!=320:raise ValueError('schedule denominator')
 return True
