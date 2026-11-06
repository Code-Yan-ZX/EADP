"""Descriptive paired score deltas; no labels, training or answer text."""
import math
from collections import Counter
import numpy as np
TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')
def compute(rows,anchors,*,draws=10000,seed=20261104):
 if len(rows)!=384 or len(anchors)!=384 or type(draws) is not int or draws<=0:raise ValueError('full denominator')
 a={(x['task'],str(x['qid'])):x for x in anchors};index={}
 if len(a)!=384 or Counter(x['task'] for x in anchors)!={t:128 for t in TASKS} or len({x['group_id'] for x in anchors})!=384:raise ValueError('anchor inventory')
 for row in rows:
  k=(row['task'],str(row['qid']))
  if k in index or k not in a or row.get('group_id')!=a[k]['group_id']:raise ValueError('paired identity')
  vals=[]
  for arm in ('rres','eadp'):
   v=row[arm]['score'];other=row[arm]['independent_score']
   if type(v) not in (int,float) or type(other) not in (int,float) or not math.isfinite(v) or not math.isfinite(other) or not 0<=v<=1 or abs(v-other)>1e-12:raise ValueError('invalid or mismatched score')
   vals.append(float(v))
  index[k]={'task':k[0],'qid':k[1],'group_id':row['group_id'],'rres_score':vals[0],'eadp_score':vals[1],'difference':vals[0]-vals[1]}
 if set(index)!=set(a):raise ValueError('missing paired score')
 rng=np.random.default_rng(seed);tasks={};per=[]
 for task in TASKS:
  subset=sorted((x for x in index.values() if x['task']==task),key=lambda r:r['group_id']);x=np.asarray([z['difference'] for z in subset]);n=len(x)
  if n!=128:raise ValueError('task denominator')
  samples=np.mean(x[rng.integers(0,n,size=(draws,n))],axis=1)
  tasks[task]={'questions':n,'image_groups':n,'rres_higher':int((x>1e-12).sum()),'ties':int((np.abs(x)<=1e-12).sum()),'eadp_higher':int((x< -1e-12).sum()),'positive_difference_sum':float(x[x>1e-12].sum()),'negative_difference_sum':float(x[x< -1e-12].sum()),'mean_difference_pp':float(x.mean()*100),'paired_bootstrap_ci95_pp':[float(z) for z in np.quantile(samples,[.025,.975])*100]}
  per.extend(subset)
 return {'status':'completed','questions':384,'tasks':tasks,'per_question':per,'bootstrap':{'unit':'image_group','replicates':draws,'seed':seed,'numpy_version':np.__version__},'GT_opened':False,'precomputed_scores_opened':True,'existing_predictions_opened':True,'predictions_used':False,'training':False,'limitations':['post-hoc DEV descriptive only','score file contains existing predictions but text is not used','continuous scores: higher/lower is not synonymous with correct/incorrect','multiple comparisons uncorrected','no causal mechanism or equal-compute claim']}
