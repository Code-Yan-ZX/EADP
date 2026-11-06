import unittest
from paired import analyze
ARMS=['stacktok_k256','eadp_k256_d1','pixel_t0','pixel_t045']
CONTRASTS={'stacktok_minus_eadp':ARMS[:2],'stacktok_minus_pixel_t0':[ARMS[0],ARMS[2]],'stacktok_minus_pixel_t045':[ARMS[0],ARMS[3]]}
class Tests(unittest.TestCase):
 def test_128_full_denominator_paired(self):
  rows=[{'group_id':str(i//2),'scores':{ARMS[0]:1.,ARMS[1]:0.,ARMS[2]:.5,ARMS[3]:1.},'answers':{ARMS[0]:'',ARMS[1]:'a',ARMS[2]:'a',ARMS[3]:''}} for i in range(128)]
  r=analyze(rows,ARMS,CONTRASTS,replicates=100,seed=20261104);self.assertEqual(r['n_questions'],128);self.assertEqual(r['n_image_groups'],64);self.assertEqual(r['contrasts']['stacktok_minus_eadp']['difference_pp'],100.);self.assertEqual(r['contrasts']['stacktok_minus_eadp']['image_cluster_percentile_95_ci_pp'],[100.,100.]);self.assertEqual(r['contrasts']['stacktok_minus_pixel_t0']['difference_pp'],50.);self.assertEqual(r['contrasts']['stacktok_minus_pixel_t045']['unchanged'],128)
 def test_reject_missing_or_nonfinite(self):
  row={'group_id':'g','scores':dict.fromkeys(ARMS,1.),'answers':dict.fromkeys(ARMS,'')};row['scores'].pop(ARMS[-1])
  with self.assertRaises(ValueError):analyze([row],ARMS,CONTRASTS,100,1)
  row['scores'][ARMS[-1]]=float('nan')
  with self.assertRaises(ValueError):analyze([row],ARMS,CONTRASTS,100,1)
if __name__=='__main__':unittest.main(verbosity=2)
