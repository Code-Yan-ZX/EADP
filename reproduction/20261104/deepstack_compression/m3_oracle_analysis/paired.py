"""Fixed seven-arm paired scores, clustered by frozen image group."""
from collections import defaultdict
import numpy as np
ARMS=('EADP','MAIN-r8','MULTI-r8','RANDOM-r8','MAIN-r16','MULTI-r16','RANDOM-r16')
CONTRASTS={'MULTI-MAIN-r8':('MULTI-r8','MAIN-r8'),'MULTI-RANDOM-r8':('MULTI-r8','RANDOM-r8'),
           'MULTI-MAIN-r16':('MULTI-r16','MAIN-r16'),'MULTI-RANDOM-r16':('MULTI-r16','RANDOM-r16')}

def excludes_minimum_gain(ci):
    return bool(ci[1]<1.0)

def analyze(rows,replicates=10000,seed=20261104):
    if not rows or type(replicates)!=int or replicates<=0:raise ValueError('invalid paired sample')
    groups=defaultdict(list)
    for row in rows:
        if set(row['scores'])!=set(ARMS) or set(row['answers'])!=set(ARMS) or not row.get('group_id'):
            raise ValueError('incomplete paired row')
        groups[row['group_id']].append(row)
    values=np.array([[float(row['scores'][arm]) for arm in ARMS] for row in rows]);ids=sorted(groups)
    if not np.isfinite(values).all() or (values<0).any() or (values>1).any():raise ValueError('invalid score')
    sums=np.array([[sum(float(row['scores'][arm]) for row in groups[g]) for arm in ARMS] for g in ids]);sizes=np.array([len(groups[g]) for g in ids])
    rng=np.random.default_rng(seed);draw=rng.integers(0,len(ids),(replicates,len(ids)))
    boot=sums[draw].sum(axis=1)/sizes[draw].sum(axis=1)[:,None]
    result={'n_questions':len(rows),'n_image_groups':len(ids),'means_pct':{arm:float(values[:,i].mean()*100) for i,arm in enumerate(ARMS)},
            'bootstrap':{'replicates':replicates,'seed':seed,'unit':'image_group'},'contrasts':{}}
    for name,(a,b) in CONTRASTS.items():
        i,j=ARMS.index(a),ARMS.index(b);diff=values[:,i]-values[:,j];ci=np.quantile((boot[:,i]-boot[:,j])*100,[.025,.975])
        result['contrasts'][name]={'difference_pp':float(diff.mean()*100),'image_group_percentile_95_ci_pp':[float(x) for x in ci],
            'improved':int((diff>1e-12).sum()),'declined':int((diff< -1e-12).sum()),'unchanged':int((abs(diff)<=1e-12).sum()),
            'answer_changed':sum(row['answers'][a]!=row['answers'][b] for row in rows),'minimum_relevant_gain_pp':1.0,
            'ci_excludes_minimum_relevant_gain':excludes_minimum_gain(ci)}
    return result
