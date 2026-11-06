import importlib.util,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT/'e4_c_latency'))
import schedule
sp=importlib.util.spec_from_file_location('independent_full_latency_review',ROOT/'e1_independent_review/review_e4c_latency_full001.py');review=importlib.util.module_from_spec(sp);sp.loader.exec_module(review)
class ScheduleReplay(unittest.TestCase):
 def test_independent_5940_job_schedule_with_exclusion(self):
  rows=[{'task':task,'qid':str(i),'group_id':task+'-'+str(i),'split':'DEV'} for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST') for i in range(128)]
  excluded={x['group_id'] for x in rows[:16]};result=review.independent_schedule(rows,excluded);self.assertEqual(result,schedule.build(rows,excluded,'full'));self.assertEqual(len(result['schedule']),5940)
  rows.pop()
  with self.assertRaises(AssertionError):review.independent_schedule(rows,excluded)
if __name__=='__main__':unittest.main()
