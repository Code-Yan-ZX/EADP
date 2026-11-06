"""Predefined E2/E3 paired summaries on already scored rows; no IO or GT."""
from collections import defaultdict
import numpy as np

CONFIGS = tuple('E2/'+x for x in ('000','100','010','001','110','101','011','111')) + ('E3/PERM','E3/MEAN')

def _difference(a, b): return {a: 1., b: -1.}

CONTRASTS = {
    'E2_leave_slot0': _difference('E2/111','E2/011'),
    'E2_leave_slot1': _difference('E2/111','E2/101'),
    'E2_leave_slot2': _difference('E2/111','E2/110'),
    'E2_only_slot0': _difference('E2/100','E2/000'),
    'E2_only_slot1': _difference('E2/010','E2/000'),
    'E2_only_slot2': _difference('E2/001','E2/000'),
    'E2_three_way': {'E2/111':1., 'E2/110':-1., 'E2/101':-1., 'E2/011':-1.,
                     'E2/100':1., 'E2/010':1., 'E2/001':1., 'E2/000':-1.},
    'E3_match_minus_perm': _difference('E2/111','E3/PERM'),
    'E3_match_minus_mean': _difference('E2/111','E3/MEAN'),
    'E3_perm_minus_off': _difference('E3/PERM','E2/000'),
    'E3_mean_minus_off': _difference('E3/MEAN','E2/000'),
}
PRIMARY = frozenset(('E2_leave_slot0','E2_leave_slot1','E2_leave_slot2'))

def analyze(rows, replicates=10000, seed=20261104):
    """Cluster bootstrap on image groups; score arrays retain question weighting.

    Primary E2 leave-one-out intervals use 1-(.05/3) coverage; other
    intervals are unadjusted exploratory 95% percentile intervals.
    """
    if not rows or type(replicates) is not int or replicates < 1:
        raise ValueError('empty rows or invalid bootstrap count')
    groups = defaultdict(list)
    for r in rows:
        if not isinstance(r.get('group_id'),str) or not r['group_id'] or set(r.get('scores',{})) != set(CONFIGS):
            raise ValueError('incomplete scores or image group')
        groups[r['group_id']].append(r)
    values=np.asarray([[float(r['scores'][k]) for k in CONFIGS] for r in rows],dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('invalid score')
    ids=sorted(groups)
    sums=np.asarray([[sum(float(r['scores'][k]) for r in groups[g]) for k in CONFIGS] for g in ids])
    sizes=np.asarray([len(groups[g]) for g in ids])
    draw=np.random.default_rng(seed).integers(0,len(ids),size=(replicates,len(ids)))
    boot=sums[draw].sum(axis=1)/sizes[draw].sum(axis=1)[:,None]
    result={'n_questions':len(rows),'n_image_groups':len(ids),'arm_means_pct':{k:float(100*values[:,i].mean()) for i,k in enumerate(CONFIGS)},
            'bootstrap':{'unit':'image_group','replicates':replicates,'seed':seed},'contrasts':{}}
    for name,weights in CONTRASTS.items():
        v=np.asarray([weights.get(k,0.) for k in CONFIGS]); observed=values@v; sampled=100*(boot@v)
        alpha=.05/3 if name in PRIMARY else .05
        interval=[float(x) for x in np.quantile(sampled,[alpha/2,1-alpha/2])]
        result['contrasts'][name]={'difference_pp':float(100*observed.mean()),'percentile_ci_pp':interval,
                                   'nominal_coverage':1-alpha,'positive_questions':int((observed>1e-12).sum()),
                                   'negative_questions':int((observed< -1e-12).sum()),'zero_questions':int((abs(observed)<=1e-12).sum())}
    return result
