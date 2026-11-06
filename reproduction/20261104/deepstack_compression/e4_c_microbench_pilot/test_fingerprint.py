import unittest,torch
from fingerprint import fingerprint
class Tests(unittest.TestCase):
 def test_full_tree_changes_are_detected(self):
  x=(torch.tensor([1.,2.],dtype=torch.bfloat16),[torch.tensor([3,4])]);y=(x[0].clone(),[x[1][0].clone()]);self.assertEqual(fingerprint(torch,x),fingerprint(torch,y));y[1][0][1]=9;self.assertNotEqual(fingerprint(torch,x),fingerprint(torch,y))
 def test_unsupported_scalar_is_rejected(self):
  with self.assertRaises(TypeError):fingerprint(torch,float('nan'))
if __name__=='__main__':unittest.main(verbosity=2)
