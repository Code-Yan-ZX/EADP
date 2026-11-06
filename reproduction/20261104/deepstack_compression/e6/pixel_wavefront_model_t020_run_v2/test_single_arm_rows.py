import copy,unittest
from single_arm import choose_row
class Tests(unittest.TestCase):
 def test_select_one_anchor_and_reject_mismatch(self):
  rows=[{'task':x,'qid':str(n),'group_id':'g'+str(n),'image_sha256':'i'+str(n),'message_sha256':'m'+str(n)} for n,x in enumerate(('D','T','A','X'))];anchor=copy.deepcopy(rows[1]);self.assertEqual(choose_row(rows,anchor),rows[1])
  bad=copy.deepcopy(anchor);bad['message_sha256']='wrong'
  with self.assertRaises(ValueError):choose_row(rows,bad)
  with self.assertRaises(ValueError):choose_row(rows+[rows[1]],anchor)
if __name__=='__main__':unittest.main(verbosity=2)
