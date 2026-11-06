import copy,json,unittest
from pathlib import Path
import verify_contract as c
D=Path(__file__).resolve().parent;FIX=D.parent/'e4_multistream_expansion352_gpu_v4/schema_fixture001.json'
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.refs=json.loads(FIX.read_text())['e1_fields']
 def test_all_nine_and_tamper_each_field(self):
  self.assertEqual(len(self.refs),9)
  for r in self.refs:
   value={k:r[k] for k in c.FIELDS};self.assertTrue(c.actual_identity(value,r))
   for k in c.FIELDS:
    altered=copy.deepcopy(value);altered[k]='tampered'
    with self.subTest(field=k),self.assertRaises(ValueError):c.actual_identity(altered,r)
 def test_missing_and_extra_fields_reject(self):
  r=self.refs[0];x={k:r[k] for k in c.FIELDS};x.pop('injected_mask')
  with self.assertRaises(ValueError):c.actual_identity(x,r)
  x={k:r[k] for k in c.FIELDS};x['extra']='x'
  with self.assertRaises(ValueError):c.actual_identity(x,r)
if __name__=='__main__':unittest.main()
