import hashlib,io,unittest
from PIL import Image
import core
class TestCore(unittest.TestCase):
 def test_boundaries(self):
  self.assertEqual(core.groups(4,5)[0],'middle');self.assertEqual(core.groups(5,4)[0],'middle');self.assertEqual(core.groups(3,5)[0],'portrait');self.assertEqual(core.groups(6,4)[0],'landscape');self.assertEqual(core.groups(999,1000)[1],'small');self.assertEqual(core.groups(1000,1000)[1],'large')
 def test_hash_and_dimensions(self):
  b=io.BytesIO();Image.new('RGB',(9,13)).save(b,format='PNG');data=b.getvalue();self.assertEqual(core.dimensions(data,hashlib.sha256(data).hexdigest()),(9,13))
  with self.assertRaises(ValueError):core.dimensions(data,'0'*64)
 def test_duplicate_and_empty(self):
  row={'task':'a','qid':'1','group_id':'g'}
  with self.assertRaises(ValueError):core.unique([row,row])
  x=core.summary([{'group_id':'g','aspect':'portrait','area':'small','rres':.5,'eadp':.2}]);self.assertEqual(x['aspect']['portrait']['difference_pp'],30.);self.assertIsNone(x['aspect']['middle']['difference_pp'])
if __name__=='__main__':unittest.main()
