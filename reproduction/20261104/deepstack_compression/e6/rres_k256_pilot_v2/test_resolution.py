import unittest
from types import SimpleNamespace
from resolution import fixed_resolution
class Tests(unittest.TestCase):
 def test_scoped_and_failure_restore(self):
  m=SimpleNamespace(QWEN3_FIXED_RESOLUTION=1024)
  with fixed_resolution(m,512):self.assertEqual(m.QWEN3_FIXED_RESOLUTION,512)
  self.assertEqual(m.QWEN3_FIXED_RESOLUTION,1024)
  with self.assertRaises(RuntimeError):
   with fixed_resolution(m,512):raise RuntimeError('test')
  self.assertEqual(m.QWEN3_FIXED_RESOLUTION,1024)
 def test_reject_foreign(self):
  for m,v in ((SimpleNamespace(QWEN3_FIXED_RESOLUTION=768),512),(SimpleNamespace(QWEN3_FIXED_RESOLUTION=1024),513)):
   with self.assertRaises(ValueError):
    with fixed_resolution(m,v):pass
if __name__=='__main__':unittest.main()
