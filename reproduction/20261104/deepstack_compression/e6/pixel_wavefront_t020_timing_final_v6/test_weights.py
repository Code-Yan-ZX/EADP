import unittest
import gpu_gate
class Tests(unittest.TestCase):
 def fixtures(self):
  exp=[{'path':str(i),'device':1,'inode':2,'size':3,'mtime_ns':4,'ctime_ns':5} for i in range(4)]
  actual=[{'path':str(i),'stat':[1,2,3,4,5]} for i in range(4)]
  return exp,actual
 def test_weight_change_and_inventory_rejected(self):
  exp,actual=self.fixtures();self.assertTrue(gpu_gate.validate_weight_stats(exp,actual))
  for bad in (actual[:-1],actual[:-1]+[dict(actual[-1],stat=[1,2,3,4,6])],actual[:-1]+[actual[0]]):
   with self.assertRaises(ValueError):gpu_gate.validate_weight_stats(exp,bad)
 def test_missing_gpu_proof_does_not_check_weights(self):
  old=gpu_gate.weight_preflight
  try:
   gpu_gate.weight_preflight=lambda:(_ for _ in ()).throw(AssertionError('should not touch files'))
   with self.assertRaises(ValueError):gpu_gate.admit()
  finally:gpu_gate.weight_preflight=old
if __name__=='__main__':unittest.main()

class RealPreflightTest(unittest.TestCase):
 def test_four_previous_weight_stats_and_review(self):
  result=gpu_gate.weight_preflight()
  self.assertEqual(len(result['expected']),4)
