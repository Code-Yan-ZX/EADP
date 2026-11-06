import os,tempfile,unittest
from pathlib import Path
from cache_contract import probe_cache
class Tests(unittest.TestCase):
 def test_real_write_and_cleanup(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'cache';result=probe_cache(path);self.assertEqual(result['path'],str(path.resolve()));self.assertTrue(result['writable']);self.assertEqual(list(path.iterdir()),[])
 def test_file_as_directory_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'file';path.write_text('x')
   with self.assertRaises((FileExistsError,NotADirectoryError,ValueError)):probe_cache(path)
if __name__=='__main__':unittest.main()
