import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'e4_c_multibudget_analysis192_v2'))
from test_metrics import synthetic
import window_detail as w
class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.jobs,cls.events=synthetic()
  for i,j in enumerate(cls.jobs):j['order']=i;cls.events[3*i+2]['record']['boundaries']['time_namespace']='time:[1]'
  cls.wins={'model_health':[[1001,1002]],'query':[[1003,1005]],'query_first_fsync':[[1003,1010]],'watcher_health':[[1007,1008]],'manual_health':[]}
 def test_full_cohort_and_unknown_manual(self):
  r=w.report(self.jobs,self.events,self.wins,'time:[1]');self.assertEqual((len(r['jobs_detail']),r['samples_excluded'],len(r['health_overlap'])),(7920,0,24));self.assertEqual(r['manual_check_evidence'],'unknown');self.assertEqual(r['jobs_detail'][0]['overlap_indices']['query']['ttft'],[0])
 def test_missing_observer_or_changed_identity(self):
  wins=copy.deepcopy(self.wins);wins['query']=[]
  with self.assertRaises(ValueError):w.report(self.jobs,self.events,wins,'time:[1]')
  events=list(self.events);i=240
  for off,key in ((1,'raw'),(2,'record')):
   x=copy.deepcopy(events[3*i+off]);x[key]['qid']='wrong';events[3*i+off]=x
  with self.assertRaises(ValueError):w.report(self.jobs,events,self.wins,'time:[1]')
 def test_incomplete_clock_and_fsync(self):
  with self.assertRaises(ValueError):w.report(self.jobs[:-1],self.events[:-3],self.wins,'time:[1]')
  with self.assertRaises(ValueError):w.report(self.jobs,self.events,self.wins,'time:[other]')
  wins=copy.deepcopy(self.wins);wins['query_first_fsync']=[[1003,1004]]
  with self.assertRaises(ValueError):w.report(self.jobs,self.events,wins,'time:[1]')
if __name__=='__main__':unittest.main()
