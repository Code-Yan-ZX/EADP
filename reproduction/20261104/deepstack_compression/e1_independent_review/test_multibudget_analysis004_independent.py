import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'e4_c_multibudget_analysis'))
from test_metrics import Tests as Fixture
import metrics
import review_e4c_multibudget_analysis004 as independent
class Tests(unittest.TestCase):
 def test_all_cells_on_1520_synthetic_jobs(self):
  jobs,events=Fixture().fixture();self.assertEqual(independent.aggregate(jobs,events),metrics.summarize(jobs,events)['tasks'])
 def test_missing_and_clock_tamper(self):
  for case in ('missing','clock','identity'):
   jobs,events=Fixture().fixture()
   if case=='missing':events.pop()
   if case=='clock':events[2]['record']['boundaries']['first_token_selected_ns']=1
   if case=='identity':events[0]['job']={'wrong':1}
   with self.subTest(case=case),self.assertRaises(ValueError):independent.aggregate(jobs,events)
 def test_exact_overlap_and_window_endpoints(self):
  self.assertEqual(independent.overlaps([[1,2],[3,4]],2,3),[0,1]);self.assertEqual(independent.overlaps([[1,2]],3,4),[])
if __name__=='__main__':unittest.main()
