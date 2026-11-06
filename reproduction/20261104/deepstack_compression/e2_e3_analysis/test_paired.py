import unittest
import numpy as np
from paired import CONFIGS, analyze

def sample(group, extra=0.):
    scores={}
    for name in CONFIGS:
        if name.startswith('E2/'):
            b=name[3:];scores[name]=.2 + .1*int(b[0]) + .05*int(b[1]) + .02*int(b[2]) + .03*(b=='111')
        else:scores[name]={'E3/PERM':.15,'E3/MEAN':.25}[name]
    scores['E2/111']+=extra
    return {'group_id':group,'scores':scores}

class PairedTests(unittest.TestCase):
    def test_fixed_factorial_effects_and_bonferroni_coverage(self):
        result=analyze([sample('a'),sample('b'),sample('c')],replicates=200,seed=9)
        contrasts=result['contrasts']
        for name,want in (('E2_leave_slot0',13.),('E2_leave_slot1',8.),('E2_leave_slot2',5.),('E2_three_way',3.)):
            self.assertAlmostEqual(contrasts[name]['difference_pp'],want)
            self.assertTrue(all(abs(x-want)<1e-9 for x in contrasts[name]['percentile_ci_pp']))
        self.assertAlmostEqual(contrasts['E2_leave_slot0']['nominal_coverage'],1-.05/3)
        self.assertAlmostEqual(contrasts['E2_three_way']['nominal_coverage'],.95)
        self.assertAlmostEqual(contrasts['E3_match_minus_perm']['difference_pp'],25.)

    def test_cluster_weighted_bootstrap_matches_manual_resampling(self):
        rows=[sample('a',.01),sample('a',.02),sample('b',.03),sample('c',.09)]
        result=analyze(rows,replicates=500,seed=17)['contrasts']['E2_leave_slot0']
        rng=np.random.default_rng(17); draws=rng.integers(0,3,size=(500,3))
        sums=np.array([.13*2+.03,.13+.03,.13+.09]);sizes=np.array([2,1,1])
        expected=100*sums[draws].sum(1)/sizes[draws].sum(1)
        self.assertTrue(np.allclose(result['percentile_ci_pp'],np.quantile(expected,[.05/6,1-.05/6])))
        self.assertAlmostEqual(result['difference_pp'],100*(.13+(.01+.02+.03+.09)/4))

    def test_missing_or_invalid_scores_fail_closed(self):
        row=sample('a');del row['scores']['E2/111']
        with self.assertRaises(ValueError):analyze([row])
        row=sample('a');row['scores']['E3/PERM']=float('nan')
        with self.assertRaises(ValueError):analyze([row])
        with self.assertRaises(ValueError):analyze([])

if __name__=='__main__':unittest.main()
