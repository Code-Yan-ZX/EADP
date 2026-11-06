"""Predefined purely numerical multi-stream diagnostic; no file I/O."""
import math
import numpy as np
N=1024
FEATURES=('main_cv','ds0_cv','ds1_cv','ds2_cv','main_ds0_rank_corr','main_ds1_rank_corr','main_ds2_rank_corr','four_stream_rank_disagreement')
TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')
def rank(values):
 values=np.asarray(values,dtype=np.float64)
 if values.ndim!=1 or not np.isfinite(values).all():raise ValueError('invalid rank inputs')
 order=np.argsort(values,kind='stable');result=np.empty(len(values),dtype=np.float64);i=0
 while i<len(values):
  j=i+1
  while j<len(values) and values[order[j]]==values[order[i]]:j+=1
  result[order[i:j]]=(i+j+1)/2;i=j
 return result
def pearson(a,b):
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
 if len(a)!=len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('invalid correlation')
 a=a-a.mean();b=b-b.mean();den=math.sqrt(float(np.dot(a,a)*np.dot(b,b)))
 return None if den==0 else float(np.dot(a,b)/den)
def compute(streams):
 if len(streams)!=4:raise ValueError('four streams required')
 ranks=[];cvs=[]
 for x in streams:
  x=np.asarray(x,dtype=np.float64)
  if x.ndim==3 and x.shape[0]==1:x=x[0]
  if x.ndim!=2 or x.shape[0]!=N or x.shape[1]<1 or not np.isfinite(x).all():raise ValueError('stream shape or values')
  norms=np.linalg.norm(x,axis=1);mean=float(norms.mean());cvs.append(None if mean==0 else float(norms.std(ddof=0)/mean));ranks.append((rank(norms)-1)/(N-1))
 out=dict(zip(FEATURES[:4],cvs))
 for i in range(3):out[FEATURES[4+i]]=pearson(ranks[0],ranks[i+1])
 out[FEATURES[7]]=float(np.var(np.stack(ranks),axis=0,ddof=0).mean())
 return out
def analyze(rows):
 expected={'DocVQA_VAL':11,'TextVQA_VAL':11,'AI2D_TEST':10}
 if len(rows)!=32 or len({(r['task'],str(r['qid'])) for r in rows})!=32 or len({(r['task'],r['group_id']) for r in rows})!=32:raise ValueError('bad cohort')
 result={}
 for task,n in expected.items():
  subset=[r for r in rows if r['task']==task]
  if len(subset)!=n:raise ValueError('task denominator')
  taskresult={}
  for name in FEATURES:
   pairs=[(r['features'][name],r['score512']-r['score128']) for r in subset if r['features'][name] is not None]
   if any(not math.isfinite(float(v)) for pair in pairs for v in pair):raise ValueError('nonfinite paired value')
   if len(pairs)<3:value=None;reason='fewer_than_three_defined_pairs'
   else:
    a,b=map(list,zip(*pairs));value=pearson(rank(a),rank(b));reason='constant_feature_or_outcome' if value is None else None
   taskresult[name]={'rho':value,'reason':reason,'n_pairs':len(pairs),'n_task':n}
  result[task]=taskresult
 return result
