import hashlib,json,struct,tempfile,unittest
from pathlib import Path
import raw_prefix
class Tests(unittest.TestCase):
 def test_prompt_bytes_and_mutation(self):
  reference={'processor':{'input_ids':{'dtype':'torch.int64','shape':[1,2],'sha256':hashlib.sha256(struct.pack('<2q',42,43)).hexdigest()}}}
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'raw.json';p.write_text(json.dumps({'full_ids':[42,43,9],'input_length':2}));self.assertEqual(raw_prefix.check(p,reference),reference['processor']['input_ids']['sha256'])
   p.write_text(json.dumps({'full_ids':[42,44,9],'input_length':2}))
   with self.assertRaises(ValueError):raw_prefix.check(p,reference)
 def test_wrong_type_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'raw.json';p.write_text(json.dumps({'full_ids':[True,2],'input_length':1}))
   with self.assertRaises(ValueError):raw_prefix.check(p,{'processor':{'input_ids':{'dtype':'torch.int64','shape':[1,1],'sha256':'x'}}})
if __name__=='__main__':unittest.main()
