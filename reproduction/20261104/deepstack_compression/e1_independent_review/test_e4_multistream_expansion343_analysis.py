import importlib.util,unittest
from pathlib import Path
import numpy as np
p=Path(__file__).with_name('review_e4_multistream_expansion343_analysis.py');s=importlib.util.spec_from_file_location('review_ms343_synthetic',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_norms_and_ties(self):
  n=np.tile(np.arange(1024,dtype=np.float64)//4,(4,1));x=m.feature(n);self.assertAlmostEqual(x['main_ds0_rank_corr'],1.);self.assertEqual(x['four_stream_rank_disagreement'],0.)
  x=m.feature(np.zeros((4,1024)));self.assertIsNone(x['main_cv']);self.assertIsNone(x['main_ds0_rank_corr'])
 def test_384_and_undefined(self):
  rows=[]
  for task in m.TASKS:
   for i in range(128):rows.append({'task':task,'qid':str(i),'group_id':task+str(i),'scores':[0.,.5,(i%7)/6],'features':{x:float(i%7) for x in m.NAMES}})
  x=m.summary(rows);self.assertEqual(x['questions'],384);self.assertAlmostEqual(x['correlations'][m.TASKS[0]]['main_cv']['rho'],1.);self.assertEqual(x['correlations'][m.TASKS[0]]['main_cv']['feature_tied_observations'],121)
  for r in rows[:128]:r['features']['main_cv']=None
  self.assertEqual(m.summary(rows)['correlations'][m.TASKS[0]]['main_cv']['feature_undefined'],128)
  with self.assertRaises(AssertionError):m.summary(rows[:-1])
if __name__=='__main__':unittest.main()
