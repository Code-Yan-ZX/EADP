import hashlib,json,tempfile,unittest
from pathlib import Path
from manifest_load import load,SEEDS
class Tests(unittest.TestCase):
 def test_fixed_full_inventory_and_tamper(self):
  rows=[{'task':'D','qid':str(i),'seed':s} for i in range(384) for s in SEEDS]
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'m';p.write_text(''.join(json.dumps(x)+'\n' for x in rows));plan={'selection_manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest()};self.assertEqual(len(load(p,plan)),1152)
   p.write_text(p.read_text()+'\n')
   with self.assertRaises(ValueError):load(p,plan)
   p.write_text(''.join(json.dumps(x)+'\n' for x in rows[:-1]+[rows[0]]));plan['selection_manifest_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
   with self.assertRaises(ValueError):load(p,plan)
if __name__=='__main__':unittest.main()
