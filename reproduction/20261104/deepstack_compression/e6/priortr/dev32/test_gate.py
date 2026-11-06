import unittest
import gate
class GateTests(unittest.TestCase):
 def test_no_execution_without_review(self):
  for preflight in (False,True):
   with self.assertRaises(ValueError):gate.execution_gate(preflight=preflight)
if __name__=='__main__':unittest.main()
