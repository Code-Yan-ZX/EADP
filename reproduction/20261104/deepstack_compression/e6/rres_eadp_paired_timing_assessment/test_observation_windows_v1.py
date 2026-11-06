"""No-GPU edge cases for pairwise time-window reporting."""
import unittest
from observation_oracle_v1 import overlap
class ObservationWindows(unittest.TestCase):
 def test_half_open_and_multiple_windows(self):
  r=overlap(10,20,[(1,10),(19,21),(20,22),(11,12)],worker_namespace='time:[1]',observer_namespace='time:[1]',coverage_start=1,coverage_end=25)
  self.assertEqual(r['matching_indices'],[1,3]);self.assertIs(r['recorded_overlap'],True);self.assertEqual(r['unobserved_interference'],'unknown')
 def test_unknown_when_namespace_or_coverage_missing(self):
  for ns,lo,hi in [('time:[2]',0,30),('time:[1]',11,30),('time:[1]',0,19),('',0,30),('time:[1]',None,30)]:
   self.assertEqual(overlap(10,20,[],worker_namespace='time:[1]',observer_namespace=ns,coverage_start=lo,coverage_end=hi)['recorded_overlap'],'unknown')
 def test_absence_from_known_windows_is_not_absence_of_interference(self):
  r=overlap(10,20,[(1,10),(20,22)],worker_namespace='time:[1]',observer_namespace='time:[1]',coverage_start=0,coverage_end=30)
  self.assertIs(r['recorded_overlap'],False);self.assertEqual(r['unobserved_interference'],'unknown')
 def test_bad_windows_and_request_rejected(self):
  for start,end,windows in [(10,10,[]),(20,10,[]),(10,20,[(8,8)]),(10,20,[(2,None)])]:
   with self.assertRaises(ValueError):overlap(start,end,windows,worker_namespace='time:[1]',observer_namespace='time:[1]',coverage_start=0,coverage_end=30)
if __name__=='__main__':unittest.main()
