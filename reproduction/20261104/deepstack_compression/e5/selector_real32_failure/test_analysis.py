import unittest,torch
from analyze import divergence
class Tests(unittest.TestCase):
 def test_common_prefix_and_exact_scores(self):
  sim=torch.eye(4);w=torch.tensor([1.,2.,3.,4.]);r=divergence(torch,sim,w,[0,1],[0,2]);self.assertEqual(r['first_divergent_step_zero_based'],1);self.assertEqual(r['set_intersection'],1);self.assertEqual(r['set_union'],3);self.assertEqual(r['cpu_float64']['author_gain'],2.);self.assertEqual(r['cpu_float64']['candidate_gain'],3.);self.assertEqual(r['cpu_float64']['author_minus_candidate'],-1.)
 def test_identical_and_duplicate_rejected(self):
  sim=torch.eye(3);w=torch.ones(3);self.assertIsNone(divergence(torch,sim,w,[0,1],[0,1])['first_divergent_step_zero_based'])
  with self.assertRaises(ValueError):divergence(torch,sim,w,[0,0],[0,1])
if __name__=='__main__':unittest.main()
