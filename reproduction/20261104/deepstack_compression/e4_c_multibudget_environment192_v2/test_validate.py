import copy,unittest
import validate
class Tests(unittest.TestCase):
 def test_roundtrip_and_bad_start(self):
  a={'event':'snapshot','index':0,'time_namespace':'time:[1]','watch_start_monotonic_ns':1,'query_start_ns':2,'query_end_ns':3,'query_start_utc':'x','query_end_utc':'x','returncode':0,'stdout':'x'}
  b={'event':'logging_complete','index':0,'time_namespace':'time:[1]','query_start_ns':2,'first_fsync_end_ns':4,'first_fsync_end_utc':'x'}
  c={'event':'watch_end','time_namespace':'time:[1]','reason':'stop_marker','samples':1,'end_ns':5,'end_utc':'x'}
  rows=[a,b,c];self.assertEqual(validate.validate(rows,'time:[1]'),[(2,3,4)])
  for key,val in [('watch_start_monotonic_ns',3),('watch_start_monotonic_ns',None),('returncode',1)]:
   bad=copy.deepcopy(rows);bad[0][key]=val
   with self.assertRaises(ValueError):validate.validate(bad,'time:[1]')
  bad=copy.deepcopy(rows);bad[-1]['reason']='duration_expired'
  with self.assertRaises(ValueError):validate.validate(bad,'time:[1]')
 def test_supervisor_health_monotonic_wrapper(self):
  import importlib.util
  p=__import__('pathlib').Path(__file__).resolve().parent.parent/'e4_c_latency/timing_supervise.py';spec=importlib.util.spec_from_file_location('_env_test_timing_supervisor',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  ticks=iter([100,200]);f=m.timed_health(lambda:{'status':'ok'},clock=lambda:next(ticks),namespace=lambda:'time:[1]');self.assertEqual(f(),{'status':'ok','health_snapshot_window_monotonic_ns':[100,200],'time_namespace':'time:[1]'})
 def test_overlap_and_unknown(self):
  self.assertEqual(validate.overlaps([(2,3,4)],3,5),{'query_indices':[0],'known_query_and_first_fsync_indices':[0],'second_fsync_overlap':'unknown'})
if __name__=='__main__':unittest.main()
