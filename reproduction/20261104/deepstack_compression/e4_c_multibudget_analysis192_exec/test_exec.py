import sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import report,analysis_gate
class Tests(unittest.TestCase):
 def test_tamper_and_cache_rejected_before_execution(self):
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp)/'gate.py';sentinel=Path(temp)/'sentinel';p.write_text('from pathlib import Path\nPath('+repr(str(sentinel))+').write_text("bad")')
   with self.assertRaisesRegex(ValueError,'unreviewed'):report.load('_fake_analysis_gate',p,'0'*64)
   self.assertFalse(sentinel.exists())
   with patch.dict(sys.modules,{'_fake_analysis_gate':object()}):
    with self.assertRaisesRegex(ValueError,'preloaded'):report.load('_fake_analysis_gate',p,report.sha(p))
 def test_missing_gate_does_not_read_target(self):
  class Denied:
   def admit(self):raise ValueError('missing complete independent proof')
  with tempfile.TemporaryDirectory() as temp,patch.object(report,'D',Path(temp)),patch.object(report,'load',return_value=Denied()),patch.object(report,'rows',side_effect=AssertionError('target opened')),patch.object(sys,'argv',['report.py','--out',str(Path(temp)/'analysis001.json')]):
   with self.assertRaisesRegex(ValueError,'missing complete'):report.main()
   self.assertFalse((Path(temp)/'analysis001.json').exists())
 def test_full_7920_combined_calculation(self):
  sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'e4_c_multibudget_analysis192_v2'));sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'e4_c_multibudget_analysis192_independent'))
  try:
   import test_metrics,metrics,window_detail
   jobs,events=test_metrics.synthetic()
   for i,j in enumerate(jobs):j['order']=i;events[3*i+2]['record']['boundaries']['time_namespace']='time:[1]'
   windows={'model_health':[[1001,1002]],'query':[[1003,1005]],'query_first_fsync':[[1003,1010]],'watcher_health':[[1007,1008]],'manual_health':[]}
   result=report.calculate(jobs,events,windows,'time:[1]',metrics,window_detail)
   self.assertEqual((len(result['jobs_detail']),len(result['health_overlap']),result['summary']['measured'],result['samples_excluded']),(7920,24,7680,0));self.assertEqual(result['manual_check_evidence'],'unknown')
  finally:del sys.path[:2]
 def test_partial_independent_pair_rejected(self):
  data={'status':'passed','GT_opened':False,'scoring_performed':False,'errors':[],'inputs':222,'generations':888,'verifier_sha256':'sha','verification_plan_sha256':'sha'}
  with patch.object(analysis_gate,'terminal'),patch.object(analysis_gate,'read',side_effect=[data,dict(data,independent_review=False)]),patch.object(analysis_gate,'sha',return_value='sha'):
   with self.assertRaisesRegex(ValueError,'not accepted'):analysis_gate.admitted_pair(*[Path('/tmp/x')]*8,{'inputs':222,'generations':888})
if __name__=='__main__':unittest.main()
