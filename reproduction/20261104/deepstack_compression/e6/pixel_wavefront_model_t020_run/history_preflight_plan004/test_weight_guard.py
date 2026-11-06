import tempfile,unittest
from pathlib import Path
from weight_guard import snapshot
class Tests(unittest.TestCase):
 def test_change_and_count(self):
  with tempfile.TemporaryDirectory() as d:
   files=[]
   for i in range(4):
    p=Path(d)/str(i);p.write_bytes(b'123');files.append({'path':str(p),'bytes':3,'sha256':'a','expected_lfs_sha256':'a'})
   before=snapshot(files);Path(files[0]['path']).write_bytes(b'4567')
   with self.assertRaises(ValueError):snapshot(files)
   with self.assertRaises(ValueError):snapshot(files[:3])
   self.assertEqual(len(before),4)
if __name__=='__main__':unittest.main(verbosity=2)
