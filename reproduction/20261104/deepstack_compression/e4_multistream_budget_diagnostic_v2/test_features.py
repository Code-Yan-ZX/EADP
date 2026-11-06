import unittest
import numpy as np
import features as f
class Tests(unittest.TestCase):
 def test_identical_and_opposite_ranks(self):
  x=np.arange(1,1025,dtype=float).reshape(1024,1);v=f.compute([x,x,x[::-1],x]);self.assertAlmostEqual(v['main_ds0_rank_corr'],1);self.assertAlmostEqual(v['main_ds1_rank_corr'],-1);self.assertGreater(v['four_stream_rank_disagreement'],0)
 def test_constant_zero_and_nan(self):
  z=np.zeros((1024,3));v=f.compute([z,z,z,z]);self.assertIsNone(v['main_cv']);self.assertIsNone(v['main_ds0_rank_corr']);self.assertEqual(v['four_stream_rank_disagreement'],0)
  bad=z.copy();bad[3,0]=float('nan')
  with self.assertRaises(ValueError):f.compute([z,z,z,bad])
  with self.assertRaises(ValueError):f.compute([z,z,z,z[:-1]])
 def test_rank_ties_and_complete_denominator(self):
  self.assertEqual(f.rank([2,1,2]).tolist(),[2.5,1,2.5])
  rows=[]
  for task,n in [('DocVQA_VAL',11),('TextVQA_VAL',11),('AI2D_TEST',10)]:
   for i in range(n):rows.append({'task':task,'qid':str(i),'group_id':str(i),'features':{k:float(i) for k in f.FEATURES},'score128':0.,'score512':i/10})
  out=f.analyze(rows);self.assertEqual(len(out['AI2D_TEST']),8);self.assertAlmostEqual(out['AI2D_TEST']['main_cv']['rho'],1)
  for r in rows:r['score512']=0.
  self.assertIsNone(f.analyze(rows)['DocVQA_VAL']['main_cv']['rho'])
  with self.assertRaises(ValueError):f.analyze(rows[:-1])
if __name__=='__main__':unittest.main()
