import unittest
from unittest import mock
import admission
class Tests(unittest.TestCase):
 def test_gpu_is_unconditionally_locked(self):
  with self.assertRaisesRegex(ValueError,'GPU locked'):admission.require_gpu()
 def test_missing_independent_proof_prevents_preflight(self):
  with mock.patch.object(admission,'read',return_value={}):
   with self.assertRaises((ValueError,FileNotFoundError)):admission.require_preflight()
if __name__=='__main__':unittest.main(verbosity=2)
