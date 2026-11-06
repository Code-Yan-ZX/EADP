import unittest,torch
from worker import imports
class TestSelector(unittest.TestCase):
 def test_boundaries_and_grids(self):
  _,A,_,candidate=imports()
  for dtype in (torch.float32,torch.bfloat16):
   t=torch.tensor(.2,dtype=dtype);v=torch.stack([torch.tensor(0.,dtype=dtype),torch.nextafter(t,torch.tensor(-float('inf'),dtype=dtype)),t,torch.nextafter(t,torch.tensor(float('inf'),dtype=dtype))]).reshape(4,1)
   for h,w in ((1,4),(4,1),(2,2)):
    self.assertTrue(torch.equal(A(method='max',threshold=.2,anchored=True)._select_2d_loco_anchored(v,h,w,v.device),candidate(v,h,w,.2)))
 def test_invalid_contract(self):
  _,_,_,f=imports()
  for args in ((torch.zeros((4,1)),2,3,.2),(torch.zeros((4,1)),2,2,0.),(torch.zeros((4,1)),2,2,float('nan'))):
   with self.assertRaises(ValueError):f(*args)
 def test_no_cuda(self):self.assertFalse(torch.cuda.is_initialized())
if __name__=='__main__':unittest.main(verbosity=2)
