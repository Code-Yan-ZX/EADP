import unittest
import preflight_gate
class Tests(unittest.TestCase):
 def test_gpu_remains_locked(self):
  with self.assertRaises(RuntimeError):preflight_gate.admit_gpu()
if __name__=='__main__':unittest.main()
