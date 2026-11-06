"""Paired factorial effects with image-group bootstrap; no scoring inputs loaded here."""
import numpy as np
from collections import defaultdict

ARMS=('P1-D0','P1-D1','P3-D0','P3-D1')
CONTRASTS={'DS_at_P1':(-1,1,0,0),'DS_at_P3':(0,0,-1,1),
           'P3_minus_P1_at_D0':(-1,0,1,0),'P3_minus_P1_at_D1':(0,-1,0,1),
           'interaction_DS_by_position':(1,-1,-1,1)}

def analyze(rows,replicates=10000,seed=20261104):
    if not rows or replicates<1:raise ValueError('empty rows or bootstrap')
    grouped=defaultdict(list)
    for row in rows:
        if set(row['scores'])!=set(ARMS):raise ValueError('incomplete score quartet')
        grouped[row['group_id']].append(row)
    groups=sorted(grouped)
    values=np.array([[float(r['scores'][a]) for a in ARMS] for r in rows],dtype=float)
    if not np.isfinite(values).all() or (values<0).any() or (values>1).any():raise ValueError('score outside [0,1]')
    group_sums=np.array([[sum(r['scores'][a] for r in grouped[g]) for a in ARMS] for g in groups],dtype=float)
    group_sizes=np.array([len(grouped[g]) for g in groups],dtype=float)
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(groups),size=(replicates,len(groups)))
    boot=group_sums[indices].sum(axis=1)/group_sizes[indices].sum(axis=1)[:,None]
    result={'n_questions':len(rows),'n_image_groups':len(groups),'replicates':replicates,'seed':seed,
            'arm_means_pct':{arm:float(100*values[:,i].mean()) for i,arm in enumerate(ARMS)},'contrasts':{}}
    for name,c in CONTRASTS.items():
        weights=np.array(c,dtype=float)
        diff=values@weights;boot_diff=100*(boot@weights)
        result['contrasts'][name]={'difference_pp':float(100*diff.mean()),
            'image_cluster_percentile_95_ci_pp':[float(x) for x in np.quantile(boot_diff,[.025,.975])],
            'improved':int((diff>1e-12).sum()),'declined':int((diff< -1e-12).sum()),
            'unchanged':int((abs(diff)<=1e-12).sum())}
    for name, a,b in [('DS_at_P1',0,1),('DS_at_P3',2,3),('P3_minus_P1_at_D0',0,2),('P3_minus_P1_at_D1',1,3)]:
        result['contrasts'][name]['answer_changed']=sum(r['answers'][ARMS[a]]!=r['answers'][ARMS[b]] for r in rows)
    return result
