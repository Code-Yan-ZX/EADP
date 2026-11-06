import unittest
from unittest.mock import patch
import smoke_gate
class Tests(unittest.TestCase):
 def test_missing_plan_refuses_both_cpu_and_gpu(self):
  with patch.object(smoke_gate,'D',smoke_gate.D/'absent_dir'):
   for preflight in (True,False):
    with self.subTest(preflight=preflight),self.assertRaisesRegex(ValueError,'plan missing'):smoke_gate.admit(preflight)
if __name__=='__main__':unittest.main()
