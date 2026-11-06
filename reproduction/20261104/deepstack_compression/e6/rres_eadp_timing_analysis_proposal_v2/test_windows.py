import unittest
import windows

def groups():return {k:{'time_namespace':'time:[123]','clock':'CLOCK_MONOTONIC','coverage_start_ns':0,'coverage_end_ns':100,'windows':[[5,10],[21,25]]} for k in windows.KINDS}
class Tests(unittest.TestCase):
 def test_recorded_overlap_and_unobserved_unknown(self):
  result=windows.annotate(9,20,'time:[123]','CLOCK_MONOTONIC',groups())
  for k in windows.KINDS:self.assertTrue(result[k]['recorded_overlap'])
  for k in windows.UNKNOWN:self.assertEqual(result[k]['recorded_overlap'],'unknown')
 def test_half_open_observation_boundary(self):
  g=groups()
  for v in g.values():v['windows']=[[5,10],[15,15],[20,21]]
  r=windows.annotate(10,20,'time:[123]','CLOCK_MONOTONIC',g)
  self.assertEqual(r['observer_query']['overlapping_windows'],[[20,21]])
 def test_no_recorded_overlap_not_no_interference(self):
  result=windows.annotate(11,20,'time:[123]','CLOCK_MONOTONIC',groups())
  self.assertFalse(result['observer_query']['recorded_overlap']);self.assertEqual(result['unobserved_host_activity']['recorded_overlap'],'unknown')
 def test_empty_mismatched_or_missing_unknown(self):
  g=groups();g['manual_check']['windows']=[];g['observer_query']['time_namespace']='other';g['run_supervisor_health']=None
  r=windows.annotate(10,20,'time:[123]','CLOCK_MONOTONIC',g)
  for k in ('manual_check','observer_query','run_supervisor_health'):self.assertEqual(r[k]['recorded_overlap'],'unknown')
 def test_outside_or_missing_coverage_unknown(self):
  g=groups();g['observer_query']['coverage_end_ns']=19;g['manual_check'].pop('coverage_start_ns')
  r=windows.annotate(11,20,'time:[123]','CLOCK_MONOTONIC',g)
  self.assertEqual(r['observer_query']['recorded_overlap'],'unknown');self.assertEqual(r['manual_check']['recorded_overlap'],'unknown')
 def test_bad_or_missing_schema_fails(self):
  g=groups();g['observer_query']['windows']=[[10,4]]
  with self.assertRaises(ValueError):windows.annotate(1,2,'time:[123]','CLOCK_MONOTONIC',g)
  g=groups();del g['manual_check']
  with self.assertRaises(ValueError):windows.annotate(1,2,'time:[123]','CLOCK_MONOTONIC',g)
