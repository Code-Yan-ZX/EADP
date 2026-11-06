import unittest,numpy as np
import core
class Tests(unittest.TestCase):
 def test_zero_and_exact_ties(self):
  x=[np.zeros((32,32,1)) for _ in range(4)];r=core.select(x,0);self.assertEqual(r['zero_streams'],4);self.assertEqual(r['tie_blocks'],256);self.assertEqual(r['ids']['minimax'],r['ids']['top_left']);self.assertTrue(all(z==0 for z in r['radii']['minimax']))
 def test_conflicting_streams_and_unique_relevant_token(self):
  x=[np.zeros((32,32,1)) for _ in range(4)];x[1][0,1,0]=0;x[1][1,0,0]=1;x[1][1,1,0]=2;r=core.select(x,0);self.assertEqual(r['block_ids']['main_minimax'][0],0);self.assertEqual(r['block_ids']['minimax'][0],32);self.assertLess(r['radii']['minimax'][0],r['radii']['main_minimax'][0]);self.assertNotIn(33,r['ids']['minimax']);self.assertEqual(r['block_ids']['minimax'][:2],[32,2]);self.assertEqual(r['ids']['minimax'],sorted(r['block_ids']['minimax']));self.assertIn(2,r['ids']['minimax'][:2])
 def test_shape_nonfinite_and_seed_determinism(self):
  x=[np.zeros((32,32,1)) for _ in range(4)];self.assertEqual(core.select(x,3),core.select(x,3));x[0][0,0,0]=np.nan
  with self.assertRaises(ValueError):core.select(x,0)
  x[0]=np.zeros((31,32,1))
  with self.assertRaises(ValueError):core.select(x,0)
 def test_finite_extremes_cannot_create_infinite_distances(self):
  x=[np.zeros((32,32,1)) for _ in range(4)];x[0][0,0,0]=7e153;x[0][0,1,0]=-7e153
  with np.errstate(over='ignore',invalid='ignore'):
   with self.assertRaisesRegex(ValueError,'nonfinite pair distance'):core.select(x,0)
 def test_tie_candidates_and_zero_stream_indices(self):
  x=[np.zeros((32,32,1)) for _ in range(4)];a=core.select(x,0);self.assertEqual(a['zero_stream_indices'],[0,1,2,3]);self.assertEqual(a['tie_cases'][0],{'block_index':0,'candidate_ids':[0,1,32,33]});self.assertEqual(len(a['tie_cases']),256)
 def test_each_radius_is_optimal(self):
  g=np.random.default_rng(91);x=[g.normal(size=(32,32,i+1)) for i in range(4)];o=core.select(x,2)
  for other in ('top_left','main_minimax',*[f'random_{s}' for s in core.SEEDS]):self.assertTrue(all(a<=b for a,b in zip(o['radii']['minimax'],o['radii'][other])))
if __name__=='__main__':unittest.main()
