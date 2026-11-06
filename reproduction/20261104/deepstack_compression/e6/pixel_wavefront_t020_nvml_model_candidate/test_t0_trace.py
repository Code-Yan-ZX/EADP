import unittest
import t0_trace
class Tensor:
 def __init__(self,values):self.values=list(values)
 def detach(self):return self
 def clone(self):return Tensor(self.values)
 def equal(self,other):return self.values==other.values
 def cpu(self):return self
 def tolist(self):return list(self.values)
class Selector:
 method='max';threshold=0;anchored=True
 def _select_2d_loco_fast(self,tokens,h,w,device):return Tensor([0,2])
class TraceTest(unittest.TestCase):
 def test_t0_calls_real_fast_and_restores(self):
  original=Selector._select_2d_loco_fast
  with t0_trace.observe(Selector,[0,2]) as seen:
   self.assertEqual(Selector()._select_2d_loco_fast(None,1,3,None).tolist(),[0,2])
  self.assertEqual(len(seen),1);self.assertIs(Selector._select_2d_loco_fast,original)
 def test_wrong_or_duplicate_indices_fail_and_cleanup(self):
  original=Selector._select_2d_loco_fast
  with self.assertRaises(ValueError):
   with t0_trace.observe(Selector,[1,2]):Selector()._select_2d_loco_fast(None,1,3,None)
  self.assertIs(Selector._select_2d_loco_fast,original)
  with self.assertRaises(ValueError):
   with t0_trace.observe(Selector,[0,2]):
    Selector()._select_2d_loco_fast(None,1,3,None);Selector()._select_2d_loco_fast(None,1,3,None)
  self.assertIs(Selector._select_2d_loco_fast,original)
