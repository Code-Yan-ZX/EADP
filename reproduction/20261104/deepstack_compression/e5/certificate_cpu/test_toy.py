"""Synthetic bounded exhaustive checks only; no real data or GPU."""
import itertools,unittest
from fractions import Fraction as F
from toy import U,MIN_NORMAL,certified_gap,gamma,round_f32,toy_gains
class TestCertificate(unittest.TestCase):
 def test_exact_ties_round_to_even(self):
  self.assertEqual(round_f32(F(1)+F(1,2**24)),F(1));self.assertEqual(round_f32(F(1)+F(3,2**24)),F(1)+F(1,2**22))
 def test_strict_boundary_and_fail_closed(self):
  E=4*gamma(5)*F(3);self.assertFalse(certified_gap(E,F(0),F(3),3,domain_proven=True));self.assertTrue(certified_gap(E+F(1,2**100),F(0),F(3),3,domain_proven=True));self.assertFalse(certified_gap(F(3),F(0),F(3),3));self.assertFalse(certified_gap(F(0),F(0),F(0),3,domain_proven=True))
 def test_underflow_and_bad_inputs_rejected(self):
  for x in (MIN_NORMAL/2,F(-1)):
   with self.assertRaises(ValueError):round_f32(x)
  with self.assertRaises(ValueError):certified_gap(F(1),F(0),F(4),3,domain_proven=True)
  with self.assertRaises(ValueError):certified_gap(1.0,F(0),F(1),3,domain_proven=True)
 def test_exhaustive_finite_toy_trees(self):
  vals=(F(0),F(1,2),F(1));rows=list(itertools.product(vals,repeat=3));coverage=(F(0),F(1,4),F(1,2));weights=(F(1),F(1,2),F(1,4));W=sum(weights);checks=cert=ties=0
  for left,right in itertools.product(rows,repeat=2):
   aa=[toy_gains(row,coverage,weights,'author') for row in (left,right)]
   for mode in ('reassociated','fma'):
    ff=[toy_gains(row,coverage,weights,mode) for row in (left,right)];best=0 if ff[0]>=ff[1] else 1;other=1-best
    if ff[0]==ff[1]:ties+=1
    accepted=certified_gap(ff[best],ff[other],W,3,domain_proven=True)
    if accepted:cert+=1;self.assertEqual(best,0 if aa[0]>=aa[1] else 1)
    checks+=1
  self.assertEqual(checks,1458);self.assertGreater(cert,0);self.assertGreater(ties,0)
 def test_1024_coefficient(self):self.assertAlmostEqual(float(4*gamma(1026)),0.00024463242249879143,places=18)
if __name__=='__main__':unittest.main(verbosity=2)
