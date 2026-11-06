import unittest
from summary import summarize
class Tests(unittest.TestCase):
 def test_unknowns_and_fully_contained_windows(self):
  bounds={'release_ns':10,'model_load_start_ns':11,'model_load_end_ns':30,'request_start_ns':31,'request_end_ns':50,'teardown_end_ns':60}
  samples=[{'query_start_ns':a,'query_end_ns':b,'gpu_uuid':'g','host_pid':42,'used_bytes':v} for a,b,v in [(1,2,None),(12,13,100),(18,19,None),(32,33,200),(49,51,300),(54,55,None)]]
  meta=[{'status':'completed','gpu_uuid':'g','query_start_ns':s['query_start_ns'],'query_end_ns':s['query_end_ns'],'device_memory_bytes':{'used':v},'processes':([] if s['used_bytes'] is None else [{'pid':42,'type':'C','used_bytes':s['used_bytes']}])} for s,v in zip(samples,[1000,1100,None,1200,1300,None])]
  r=summarize(samples,meta,bounds,20)['stages'];self.assertEqual(r['pre_cuda']['process'],{'known':0,'unknown':1,'max_known_bytes':None,'present':0,'absent':1,'present_unknown':0});self.assertEqual(r['model_load']['process'],{'known':1,'unknown':1,'max_known_bytes':100,'present':1,'absent':1,'present_unknown':0});self.assertEqual(r['request']['process']['max_known_bytes'],200);self.assertEqual(r['request']['fully_contained_samples'],1);self.assertIsNone(r['teardown']['device_used']['max_known_bytes'])
 def test_unknown_target_process_is_not_zero(self):
  b={'release_ns':10,'model_load_start_ns':11,'model_load_end_ns':20,'request_start_ns':21,'request_end_ns':30,'teardown_end_ns':40}
  s={'query_start_ns':12,'query_end_ns':13,'gpu_uuid':'g','host_pid':42,'used_bytes':None}
  m={'status':'completed','gpu_uuid':'g','query_start_ns':12,'query_end_ns':13,'device_memory_bytes':{'used':None},'processes':[{'pid':42,'type':'C','used_bytes':None}]}
  process=summarize([s],[m],b,20)['stages']['model_load']['process'];self.assertEqual(process['present_unknown'],1);self.assertEqual(process['unknown'],1);self.assertIsNone(process['max_known_bytes'])
 def test_invalid_clock_and_missing_samples_fail(self):
  b={'release_ns':1,'model_load_start_ns':2,'model_load_end_ns':3,'request_start_ns':4,'request_end_ns':5,'teardown_end_ns':6}
  with self.assertRaises(ValueError):summarize([],[],b,20)
  b['request_end_ns']=2
  with self.assertRaises(ValueError):summarize([{}],[{}],b,20)
if __name__=='__main__':unittest.main(verbosity=2)
