"""Fixed K/DS within-image paired summary; no external data access."""
import numpy as np
from collections import defaultdict

def analyze(rows,configs,contrasts,replicates=10000,seed=20261104):
 if not rows or replicates<1 or len(configs)!=8 or len(set(configs))!=8:raise ValueError('bad inventory')
 groups=defaultdict(list)
 for r in rows:
  if set(r['scores'])!=set(configs) or set(r['answers'])!=set(configs):raise ValueError('incomplete configuration')
  groups[r['group_id']].append(r)
 ids=sorted(groups);values=np.array([[float(r['scores'][c]) for c in configs] for r in rows])
 if not np.isfinite(values).all() or (values<0).any() or (values>1).any():raise ValueError('score outside [0,1]')
 sums=np.array([[sum(float(x['scores'][c]) for x in groups[g]) for c in configs] for g in ids]);sizes=np.array([len(groups[g]) for g in ids]);rng=np.random.default_rng(seed)
 draw=rng.integers(0,len(ids),(replicates,len(ids)));boot=sums[draw].sum(axis=1)/sizes[draw].sum(axis=1)[:,None]
 result={'n_questions':len(rows),'n_image_groups':len(ids),'arm_means_pct':{c:float(100*values[:,i].mean()) for i,c in enumerate(configs)},'bootstrap':{'replicates':replicates,'seed':seed,'unit':'image_group'},'contrasts':{}}
 result['arm_percentile_95_ci_pct']={c:[float(x) for x in np.quantile(100*boot[:,i],[.025,.975])] for i,c in enumerate(configs)}
 result['quality_curves']={mode:[{'budget':k,'mean_pct':result['arm_means_pct']['K'+str(k)+'-'+mode],
    'image_cluster_percentile_95_ci_pct':result['arm_percentile_95_ci_pct']['K'+str(k)+'-'+mode]} for k in (128,256,512,1024)] for mode in ('D0','D1')}
 for name,(a,b) in contrasts.items():
  if a not in configs or b not in configs or a==b:raise ValueError('bad paired contrast')
  ai=configs.index(a);bi=configs.index(b);diff=values[:,ai]-values[:,bi];boot_diff=100*(boot[:,ai]-boot[:,bi])
  result['contrasts'][name]={'difference_pp':float(100*diff.mean()),'image_cluster_percentile_95_ci_pp':[float(x) for x in np.quantile(boot_diff,[.025,.975])],
    'improved':int((diff>1e-12).sum()),'declined':int((diff< -1e-12).sum()),'unchanged':int((abs(diff)<=1e-12).sum()),'answer_changed':sum(r['answers'][a]!=r['answers'][b] for r in rows)}
 return result
