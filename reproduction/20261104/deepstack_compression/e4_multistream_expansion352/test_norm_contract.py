import importlib.util,sys,unittest
from pathlib import Path
import numpy as np
import norm_contract as n
class NormContractTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.f=n.original()
 def test_exact_reconstruction(self):
  rng=np.random.default_rng(20261105);streams=[rng.standard_normal((1024,k)).astype(np.float32) for k in (3,4,5,6)];self.assertEqual(self.f.compute(streams),n.features_from_norms(n.norms(streams),self.f))
 def test_constant_and_zero(self):
  streams=[np.zeros((1024,2),dtype=np.float64) for _ in range(4)];self.assertEqual(self.f.compute(streams),n.features_from_norms(n.norms(streams),self.f))
 def test_bad_shape_dtype_nonfinite(self):
  for data in (np.zeros((3,1024)),np.zeros((4,1024),dtype=np.float32),np.full((4,1024),np.nan),-np.ones((4,1024))):
   with self.assertRaises(ValueError):n.features_from_norms(data,self.f)
 def test_original_tamper_or_cache(self):
  with self.assertRaises(ValueError):n.original()
if __name__=='__main__':unittest.main()
