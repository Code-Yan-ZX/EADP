import tempfile,unittest
from pathlib import Path
from snapshot_sources import sha,snapshot
class Tests(unittest.TestCase):
 def test_copy_and_changed_source_fail(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'source';p.write_bytes(b'bytes');sources={str(p):sha(p)};items=snapshot(sources,Path(d)/'snap');self.assertEqual(sha(items[str(p)]['path']),sources[str(p)])
   p.write_bytes(b'changed')
   with self.assertRaisesRegex(ValueError,'changed source'):snapshot(sources,Path(d)/'bad')
if __name__=='__main__':unittest.main(verbosity=2)
