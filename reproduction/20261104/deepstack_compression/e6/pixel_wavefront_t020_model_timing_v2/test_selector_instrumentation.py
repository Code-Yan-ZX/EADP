import unittest
from selector_instrumentation import observe
class Tensor:
 def __init__(self,ids):self.ids=ids;self._version=0
 def detach(self):return self
 def clone(self):return Tensor(list(self.ids))
 def equal(self,other):return self.ids==other.ids
 def cpu(self):return self
 def tolist(self):return self.ids
class Selector:
 method='max';threshold=.2;anchored=True
 def _select_2d_loco_anchored(self,t,h,w,d):return Tensor([1,4])
class Tests(unittest.TestCase):
 def test_both_paths_and_cleanup(self):
  old=Selector._select_2d_loco_anchored
  for arm in ('author','wavefront'):
   with observe(Selector,lambda *args:Tensor([1,4]),arm,[1,4]):self.assertEqual(Selector()._select_2d_loco_anchored(None,2,2,None).tolist(),[1,4])
   self.assertIs(Selector._select_2d_loco_anchored,old)
 def test_real_inference_tensor_and_mutation(self):
  import torch
  class Actual:
   method='max';threshold=.2;anchored=True
   def _select_2d_loco_anchored(self,*args):return torch.tensor([1,4])
  with torch.inference_mode():
   with observe(Actual,None,'author',[1,4]):Actual()._select_2d_loco_anchored(None,2,2,None)
   with self.assertRaises(ValueError):
    with observe(Actual,None,'author',[1,4]):x=Actual()._select_2d_loco_anchored(None,2,2,None);x[0]=0
 def test_mismatch_mutation_and_exception_restore(self):
  old=Selector._select_2d_loco_anchored
  with self.assertRaises(ValueError):
   with observe(Selector,None,'author',[1,9]):Selector()._select_2d_loco_anchored(None,2,2,None)
  self.assertIs(Selector._select_2d_loco_anchored,old)
  with self.assertRaises(ValueError):
   with observe(Selector,None,'author',[1,4]):x=Selector()._select_2d_loco_anchored(None,2,2,None);x.ids[0]=9
  self.assertIs(Selector._select_2d_loco_anchored,old)
  with self.assertRaises(RuntimeError):
   with observe(Selector,None,'author',[1,4]):raise RuntimeError('x')
  self.assertIs(Selector._select_2d_loco_anchored,old)
if __name__=='__main__':unittest.main()
