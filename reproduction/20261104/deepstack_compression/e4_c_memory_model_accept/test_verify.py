import unittest
from verify import independent_windows
class Tests(unittest.TestCase):
 def test_unknown_and_overlap_not_counted(self):
  bounds={'release_ns':10,'model_load_start_ns':11,'model_load_end_ns':30,'request_start_ns':31,'request_end_ns':50,'teardown_end_ns':60}
  pairs=[(1,2,None),(12,13,100),(18,19,None),(32,33,200),(49,51,300),(54,55,None)];s=[{'query_start_ns':a,'query_end_ns':b,'gpu_uuid':'g','host_pid':42,'used_bytes':v} for a,b,v in pairs];m=[{'gpu_uuid':'g','query_start_ns':x['query_start_ns'],'query_end_ns':x['query_end_ns'],'device_memory_bytes':{'used':v},'processes':([] if x['used_bytes'] is None else [{'pid':42,'type':'C','used_bytes':x['used_bytes']}])} for x,v in zip(s,[1000,1100,None,1200,1300,None])]
  r=independent_windows(s,m,bounds,20);self.assertIsNone(r['pre_cuda']['process']['max_known_bytes']);self.assertEqual(r['model_load']['process']['max_known_bytes'],100);self.assertEqual(r['request']['fully_contained_samples'],1);self.assertEqual(r['request']['device_used']['max_known_bytes'],1200)
  m[1]['processes'][0]['used_bytes']=999
  with self.assertRaises(ValueError):independent_windows(s,m,bounds,20)
if __name__=='__main__':unittest.main(verbosity=2)
