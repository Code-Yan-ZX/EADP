"""Fixed 384-row descriptive contract; pure CPU, no file or label access."""
import math
import numpy as np
TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')
FEATURES=('main_cv','ds0_cv','ds1_cv','ds2_cv','main_ds0_rank_corr','main_ds1_rank_corr','main_ds2_rank_corr','four_stream_rank_disagreement')
def accepted_input_identity(row,chosen):
 return row.get('split','DEV')=='DEV' and all(str(row.get(k))==str(chosen[k]) for k in ('task','qid','group_id','image_sha256','message_sha256'))
def rank(x):
 a=np.asarray(x,dtype=np.float64)
 if a.ndim!=1 or not np.isfinite(a).all():raise ValueError('invalid ranks')
 order=np.argsort(a,kind='stable');r=np.empty(len(a),dtype=np.float64);i=0
 while i<len(a):
  j=i+1
  while j<len(a) and a[order[j]]==a[order[i]]:j+=1
  r[order[i:j]]=(i+j+1)/2;i=j
 return r
def corr(x,y):
 a=np.asarray(x,dtype=np.float64);b=np.asarray(y,dtype=np.float64);a-=a.mean();b-=b.mean();d=math.sqrt(float(np.dot(a,a)*np.dot(b,b)))
 return None if d==0 else float(np.dot(a,b)/d)
def norm_features(n):
 n=np.asarray(n)
 if n.shape!=(4,1024) or n.dtype!=np.float64 or not np.isfinite(n).all() or (n<0).any():raise ValueError('bad norm array')
 r=[(rank(x)-1)/1023 for x in n];out={}
 for i in range(4):
  m=float(n[i].mean());out[FEATURES[i]]=None if m==0 else float(n[i].std(ddof=0)/m)
 for i in range(3):out[FEATURES[4+i]]=corr(r[0].copy(),r[i+1].copy())
 out[FEATURES[7]]=float(np.var(np.stack(r),axis=0,ddof=0).mean());return out
def summarize(rows):
 if len(rows)!=384 or len({(r['task'],str(r['qid'])) for r in rows})!=384:raise ValueError('incomplete or duplicate panel')
 if set(r['task'] for r in rows)!=set(TASKS):raise ValueError('unexpected task')
 result={};groups={}
 for task in TASKS:
  taskrows=[r for r in rows if r['task']==task];groups[task]={r['group_id'] for r in taskrows}
  if len(taskrows)!=128 or len(groups[task])!=128:raise ValueError('invalid task group denominator')
  for r in taskrows:
   if set(r['features'])!=set(FEATURES):raise ValueError('feature inventory')
   for x in r['scores']:
    if type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1:raise ValueError('invalid score')
   if len(r['scores'])!=3:raise ValueError('score inventory')
   for x in r['features'].values():
    if x is not None and (type(x) not in (int,float) or not math.isfinite(x)):raise ValueError('invalid feature')
  out={}
  for name in FEATURES:
   pairs=[(r['features'][name],r['scores'][2]-r['scores'][0]) for r in taskrows if r['features'][name] is not None];n=len(pairs)
   x=[p[0] for p in pairs];y=[p[1] for p in pairs]
   reason='fewer_than_three_defined_pairs' if n<3 else ('constant_feature' if len(set(x))==1 else ('constant_outcome' if len(set(y))==1 else None))
   out[name]={'rho':None if reason else corr(rank(x),rank(y)),'reason':reason,'n_task':128,'feature_defined':n,'feature_undefined':128-n,'outcome_defined':128,'pairs':n,'feature_tied_observations':n-len(set(x)),'outcome_tied_observations':n-len(set(y))}
  result[task]=out
 overlap={a+'|'+b:len(groups[a]&groups[b]) for i,a in enumerate(TASKS) for b in TASKS[i+1:]}
 return {'questions':384,'per_task':128,'correlations':result,'cross_task_group_overlap':overlap}
