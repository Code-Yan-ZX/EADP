import tempfile,unittest
from pathlib import Path
from freeze_plan import source_key
class Paths(unittest.TestCase):
 def test_model_alias_remains_container_visible(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);actual=root/'host_blob';actual.write_text('x');alias=root/'mounted'/'model';alias.parent.mkdir();alias.symlink_to(actual)
   self.assertEqual(source_key(alias),str(alias.absolute()))
   self.assertNotEqual(source_key(alias),str(alias.resolve()))
   self.assertEqual(Path(source_key(alias)).read_text(),'x')
if __name__=='__main__':unittest.main()
