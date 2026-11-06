import copy,unittest
from contract import BUDGETS,schedule,validate
class Tests(unittest.TestCase):
 def rows(self):return [{'task':'task'+str(i),'qid':str(i),'group_id':'g'+str(i),'split':'DEV'} for i in range(3)]
 def test_full_randomized_paired_blocks(self):
  rows=self.rows();jobs=schedule(rows);self.assertTrue(validate(jobs,rows));self.assertEqual(sum(x['phase']=='warmup' for x in jobs),12);self.assertEqual(sum(x['phase']=='measure' for x in jobs),36)
  for index in range(0,len(jobs),4):
   block=jobs[index:index+4];self.assertEqual({x['budget'] for x in block},set(BUDGETS));self.assertEqual(len({(x['task'],x['phase'],x['replicate']) for x in block}),1)
 def test_missing_or_corrupt_order_rejected(self):
  rows=self.rows();jobs=schedule(rows)
  with self.assertRaises(ValueError):validate(jobs[:-1],rows)
  bad=copy.deepcopy(jobs);bad[0]['budget']=999
  with self.assertRaises(ValueError):validate(bad,rows)
 def test_reject_wrong_split_or_duplicate_image(self):
  rows=self.rows();rows[0]['split']='FINAL'
  with self.assertRaises(ValueError):schedule(rows)
  rows=self.rows();rows[0]['group_id']=rows[1]['group_id']
  with self.assertRaises(ValueError):schedule(rows)
