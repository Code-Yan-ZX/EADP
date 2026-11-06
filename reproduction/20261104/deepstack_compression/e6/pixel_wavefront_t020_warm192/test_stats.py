import copy,unittest
import stats
class StatsTest(unittest.TestCase):
 def fixture(self):
  block=[];records=[]
  for task in stats.TASKS:
   for i in range(3):
    row={'task':task,'qid':str(i),'group_id':task+str(i)};block.append(row)
    for arm,delta in [('author',10+i),('wavefront',12+i)]:records.append(dict(row,arm=arm,boundaries={'t0_before_image_read_ns':100,'first_token_selected_ns':100+delta},first_token_id=9 if arm=='author' else 10,selected_indices=[i]))
  return {'blocks':[block]},records
 def test_mismatch_kept_and_bootstrap_reproducible(self):
  d,r=self.fixture();a=stats.calculate(d,r,draws=31,expected_per_task=3);b=stats.calculate(d,r,draws=31,expected_per_task=3);self.assertEqual(a,b);self.assertEqual(a['observations'],18);self.assertEqual(a['tasks']['DocVQA_VAL']['first_token_mismatches'],3);self.assertEqual(a['tasks']['TextVQA_VAL']['paired_difference_median_ns'],2)
 def test_missing_duplicate_or_nonpositive_rejected(self):
  d,r=self.fixture()
  with self.assertRaises(ValueError):stats.calculate(d,r[:-1],draws=2,expected_per_task=3)
  r[-1]['boundaries']['first_token_selected_ns']=100
  with self.assertRaises(ValueError):stats.calculate(d,r,draws=2,expected_per_task=3)
if __name__=='__main__':unittest.main()
