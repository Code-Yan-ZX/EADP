"""Question-weighted image-group paired bootstrap, with fixed random seeds."""
from collections import defaultdict
import numpy as np

SEEDS=(20261104,20261105,20261106)
ARMS=('rres','eadp','e1_full','random_20261104','random_20261105','random_20261106','random_three_seed_mean')
def analyze(rows,replicates=10000,seed=20261104):
 if not rows or type(replicates) is not int or replicates<1:raise ValueError('invalid sample')
 groups=defaultdict(list);seen=set()
 for row in rows:
  qid=row.get('qid');group=row.get('group_id');scores=row.get('scores')
  if not isinstance(qid,str) or not qid or qid in seen or not isinstance(group,str) or not group:raise ValueError('question/group identity')
  if not isinstance(scores,dict) or set(scores)!={'rres','eadp','e1_full',*SEEDS}:raise ValueError('arm inventory')
  seen.add(qid);v=[float(scores['rres']),float(scores['eadp']),float(scores['e1_full'])]+[float(scores[s]) for s in SEEDS]
  if any(not np.isfinite(x) or not 0<=x<=1 for x in v):raise ValueError('invalid score')
  v.append(sum(v[3:])/3);groups[group].append(v)
 ids=sorted(groups);sums=np.asarray([np.asarray(groups[g]).sum(axis=0) for g in ids]);sizes=np.asarray([len(groups[g]) for g in ids])
 mean=100*sums.sum(axis=0)/sizes.sum();draw=np.random.default_rng(seed).integers(0,len(ids),(replicates,len(ids)));boot=100*sums[draw].sum(axis=1)/sizes[draw].sum(axis=1)[:,None]
 ci=lambda x:[float(z) for z in np.quantile(x,[.025,.975])]
 return {'questions':len(rows),'image_groups':len(ids),'arm_mean_pct':{a:float(mean[i]) for i,a in enumerate(ARMS)},
         'arm_ci95_pct':{a:ci(boot[:,i]) for i,a in enumerate(ARMS)},
         'rres_minus':{a:{'difference_pp':float(mean[0]-mean[i]),'ci95_pp':ci(boot[:,0]-boot[:,i])} for i,a in enumerate(ARMS) if i},
         'bootstrap':{'unit':'image_group','group_order':'ascending','weighting':'question','replicates':replicates,'seed':seed,'numpy_version':np.__version__}}
