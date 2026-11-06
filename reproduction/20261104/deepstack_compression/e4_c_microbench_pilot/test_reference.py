import unittest
from reference_contract import choose,FIELDS
class Tests(unittest.TestCase):
 def test_unique_dev_identity_only(self):
  row={'task':'D','qid':'1','group_id':'g','message_sha256':'m'};r={k:None for k in FIELDS};r.update(row,arm='P3-D1',forced=False,answer='hidden',output_ids=[17]);out=choose([r],row)
  self.assertEqual(set(out),set(FIELDS));self.assertNotIn('answer',out);self.assertNotIn('output_ids',out)
  with self.assertRaises(ValueError):choose([r,r],row)
  with self.assertRaises(ValueError):choose([{**r,'group_id':'other'}],row)
  with self.assertRaises(ValueError):choose([{**r,'forced':True}],row)
if __name__=='__main__':unittest.main(verbosity=2)
