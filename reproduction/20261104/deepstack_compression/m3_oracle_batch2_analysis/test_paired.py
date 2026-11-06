import unittest
import paired
class PairedTests(unittest.TestCase):
    def test_fixed_group_bootstrap_and_pairs(self):
        rows=[]
        for group in ('a','b','c','d'):
            scores={a:0. for a in paired.ARMS};scores['MULTI-r8']=0.5;scores['MULTI-r16']=0.25
            rows.append({'group_id':group,'scores':scores,'answers':{a:'' for a in paired.ARMS}})
        result=paired.analyze(rows,replicates=100,seed=5)
        self.assertEqual(result['contrasts']['MULTI-MAIN-r8']['difference_pp'],50.)
        self.assertEqual(result['contrasts']['MULTI-RANDOM-r16']['image_group_percentile_95_ci_pp'],[25.,25.])
        self.assertEqual(result['n_questions'],4)
        self.assertTrue(paired.excludes_minimum_gain([0.,0.999]));self.assertFalse(paired.excludes_minimum_gain([0.,1.0]))
        self.assertFalse(result['contrasts']['MULTI-MAIN-r8']['ci_excludes_minimum_relevant_gain'])
        for row in rows:row['scores']['MULTI-r8']=0.
        self.assertTrue(paired.analyze(rows,replicates=10)['contrasts']['MULTI-MAIN-r8']['ci_excludes_minimum_relevant_gain'])
        del rows[0]['scores']['EADP']
        with self.assertRaises(ValueError):paired.analyze(rows,replicates=10)
if __name__=='__main__':unittest.main()
