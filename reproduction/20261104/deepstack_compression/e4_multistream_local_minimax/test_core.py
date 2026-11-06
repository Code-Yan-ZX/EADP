import unittest,numpy as np
import core
class Tests(unittest.TestCase):
 def test_zero_and_exact_ties(self):
  x=[np.zeros((32,32,1)) for _ in range(4)];r=core.select(x,0);self.assertEqual(r['zero_streams'],4);self.assertEqual(r['tie_blocks'],256);self.assertEqual(r['ids']['minimax'],r['ids']['top_left']);self.assertTrue(all(z==0 for z in r['radii']['minimax']))
 def test_conflicting_streams_and_unique_relevant_token(self):
  x=[np.zeros((32,32,1)) for _ in range(4)];x[1][0,1,0]=0;x[1][1,0,0]=1;x[1][1,1,0]=2;r=core.select(x,0);self.assertEqual(r['ids']['main_minimax'][0],0);self.assertEqual(r['ids']['minimax'][0],32);self.assertLess(r['radii']['minimax'][0],r['radii']['main_minimax'][0]);self.assertNotIn(33,r['ids']['minimax'])
 def test_shape_nonfinite_and_seed_determinism(self):
  x=[np.zeros((32,32,1)) for _ in range(4)];self.assertEqual(core.select(x,3),core.select(x,3));x[0][0,0,0]=np.nan
  with self.assertRaises(ValueError):core.select(x,0)
  x[0]=np.zeros((31,32,1))
  with self.assertRaises(ValueError):core.select(x,0)
 def test_each_radius_is_optimal(self):
  g=np.random.default_rng(91);x=[g.normal(size=(32,32,i+1)) for i in range(4)];o=core.select(x,2)
  for other in ('top_left','main_minimax',*[f'random_{s}' for s in core.SEEDS]):self.assertTrue(all(a<=b for a,b in zip(o['radii']['minimax'],o['radii'][other])))
if __name__=='__main__':unittest.main()
