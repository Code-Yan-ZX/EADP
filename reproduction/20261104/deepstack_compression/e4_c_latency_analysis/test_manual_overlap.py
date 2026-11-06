import unittest
from manual_overlap import validate_windows,annotate
class ManualOverlapTest(unittest.TestCase):
 def record(self,a,z,phase='measure'):
  return {'status':'generated','job':{'phase':phase},'result':{'boundaries':{'t0_before_image_read_ns':a,'generation_complete_ns':z,'time_namespace':'time:1','clock_implementation':'clock_gettime(CLOCK_MONOTONIC)'}}}
 def test_inclusive_and_null_nvml(self):
  windows=validate_windows([{'time_namespace':'time:1','manual_health_window_monotonic_ns':[10,20],'nvml_window_monotonic_ns':[15,18]},{'time_namespace':'time:1','manual_health_window_monotonic_ns':[40,50],'nvml_window_monotonic_ns':None,'gpu_check':'reused; no additional NVML query'}],'time:1')
  rows=annotate([self.record(5,10),self.record(18,19),self.record(50,51),self.record(21,39)],windows,'time:1');self.assertEqual(len(rows),4)
  self.assertEqual([x['manual_health_overlap'] for x in rows],[True,True,True,False]);self.assertEqual([x['manual_nvml_overlap'] for x in rows],[False,True,False,False])
 def test_unknown_or_outside_fails_closed(self):
  for change in ({'time_namespace':'wrong'},{'nvml_window_monotonic_ns':None},{'nvml_window_monotonic_ns':[0,11]}, {'manual_health_window_monotonic_ns':[20,10]}):
   x={'time_namespace':'time:1','manual_health_window_monotonic_ns':[10,20],'nvml_window_monotonic_ns':[11,12]};x.update(change)
   with self.assertRaises(ValueError):validate_windows([x],'time:1')
 def test_incompatible_worker_clock_rejected(self):
  w=validate_windows([{'time_namespace':'time:1','manual_health_window_monotonic_ns':[1,2],'nvml_window_monotonic_ns':[1,2]}],'time:1');r=self.record(1,2);r['result']['boundaries']['time_namespace']='other'
  with self.assertRaises(ValueError):annotate([r],w,'time:1')
if __name__=='__main__':unittest.main()
