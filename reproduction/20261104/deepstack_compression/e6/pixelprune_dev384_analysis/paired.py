"""Fixed four-configuration DEV summaries; no data access on import."""
from collections import defaultdict
import numpy as np

def analyze(rows,configs,contrasts,replicates=10000,seed=20261104):
 if not rows or len(configs)!=4 or len(set(configs))!=4 or type(replicates) is not int or replicates<1:raise ValueError('bad paired inventory')
 groups=defaultdict(list)
 for row in rows:
  if not isinstance(row.get('group_id'),str) or not row['group_id'] or set(row['scores'])!=set(configs) or set(row['answers'])!=set(configs):raise ValueError('incomplete paired row')
  if any(not isinstance(row['answers'][c],str) for c in configs):raise ValueError('non-string answer')
  groups[row['group_id']].append(row)
 ids=sorted(groups);values=np.array([[float(r['scores'][c]) for c in configs] for r in rows],dtype=float)
 if not np.isfinite(values).all() or (values<0).any() or (values>1).any():raise ValueError('score outside [0,1]')
 sums=np.array([[sum(float(x['scores'][c]) for x in groups[g]) for c in configs] for g in ids]);sizes=np.array([len(groups[g]) for g in ids]);rng=np.random.default_rng(seed)
 draw=rng.integers(0,len(ids),(replicates,len(ids)));boots=sums[draw].sum(axis=1)/sizes[draw].sum(axis=1)[:,None]
 out={'n_questions':len(rows),'n_image_groups':len(ids),'arm_means_pct':{c:float(100*values[:,i].mean()) for i,c in enumerate(configs)},
      'arm_percentile_95_ci_pct':{c:[float(v) for v in np.quantile(100*boots[:,i],[.025,.975])] for i,c in enumerate(configs)},
      'bootstrap':{'unit':'image_group','replicates':replicates,'seed':seed,'method':'percentile'},'contrasts':{}}
 for name,pair in contrasts.items():
  if len(pair)!=2 or pair[0] not in configs or pair[1] not in configs or pair[0]==pair[1]:raise ValueError('invalid contrast')
  a,b=pair;i=configs.index(a);j=configs.index(b);d=values[:,i]-values[:,j];bd=100*(boots[:,i]-boots[:,j])
  out['contrasts'][name]={'difference_pp':float(100*d.mean()),'image_cluster_percentile_95_ci_pp':[float(v) for v in np.quantile(bd,[.025,.975])],
       'improved':int((d>1e-12).sum()),'declined':int((d< -1e-12).sum()),'unchanged':int((abs(d)<=1e-12).sum()),'answer_changed':sum(r['answers'][a]!=r['answers'][b] for r in rows)}
 return out
