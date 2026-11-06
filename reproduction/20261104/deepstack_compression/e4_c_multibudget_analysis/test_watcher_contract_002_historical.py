import copy, unittest
import watcher_contract as w
T='2026-11-05T00:00:00+00:00';NS='time:[1]'
def fixture():
 return ([{'event':'snapshot','index':0,'time_namespace':NS,'query_start_ns':10,'query_end_ns':20,'query_start_utc':T,'query_end_utc':T,'returncode':0,'stdout':''},{'event':'logging_complete','index':0,'time_namespace':NS,'query_start_ns':10,'first_fsync_end_ns':30,'first_fsync_end_utc':T},{'event':'watch_end','time_namespace':NS,'reason':'stop_marker','samples':1,'end_ns':40,'end_utc':T}], [{'time_namespace':NS,'health_snapshot_window_monotonic_ns':[12,14]}],[{'event':'manual_health_check','time_namespace':NS,'start_monotonic_ns':13,'end_monotonic_ns':15,'start_utc':T,'end_utc':T,'details':{}}])
class Tests(unittest.TestCase):
 def test_valid_and_overlap(self):
  result=w.validate(*fixture(),NS,T,T);self.assertEqual(w.overlap(result['query'],19,25),[0]);self.assertEqual(w.overlap(result['manual_health'],20,25),[])
 def test_invalid_or_missing_evidence_fails(self):
  for variant in ('missing','failed','namespace','truncated','manual','health','stop','coverage'):
   s,h,m=copy.deepcopy(fixture())
   if variant=='missing':s=[]
   if variant=='failed':s[0]['returncode']=1
   if variant=='namespace':s[0]['time_namespace']='other'
   if variant=='truncated':s.pop(1)
   if variant=='manual':m=[]
   if variant=='health':h=[]
   if variant=='stop':s[-1]['reason']='duration'
   end='2026-11-05T00:00:01+00:00' if variant=='coverage' else T
   with self.subTest(variant=variant),self.assertRaises(ValueError):w.validate(s,h,m,NS,T,end)
 def test_hourly_missing_is_unknown(self):
  span=3600*10**9;r=w.hourly_coverage(0,2*span,[[span-1,span+1]],[]);self.assertEqual([x['status'] for x in r],['observed','unknown'])
if __name__=='__main__':unittest.main()
