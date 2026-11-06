import unittest
import nvml_stages
M=1048576
class Tests(unittest.TestCase):
 def samples(self):
  return [{'status':'completed','gpu_uuid':'u','query_start_ns':a,'query_end_ns':b,'device_memory_bytes':{'total':20*M,'used':10*M,'free':10*M,'reserved':None},'processes':([{'pid':7,'type':'C','used_bytes':4*M}] if i else [])} for i,(a,b) in enumerate([(5,12),(14,17),(19,22),(24,27),(29,32),(34,37),(39,42)])]
 def test_unknown_and_boundary_overlap(self):
  r=nvml_stages.summarize(self.samples(),7,'u',{'model_load_start':10,'model_load_end':20,'request_start':21,'request_end':30,'teardown_end':40},15)
  self.assertEqual(r['stages']['model_load']['overlap_indices'],[0,1,2]);self.assertEqual(r['stages']['model_load']['target_absent_samples'],1)
  self.assertEqual(r['stages']['model_load']['metrics']['target_used_bytes']['unknown_samples'],1)
  self.assertEqual(r['stages']['model_load']['metrics']['target_used_bytes']['known_sample_max_bytes'],4*M)
  self.assertIsNone(r['stages']['request']['metrics']['reserved']['known_sample_max_bytes'])
 def test_wrong_stage_or_missing_bracket_rejected(self):
  with self.assertRaises(ValueError):nvml_stages.summarize(self.samples(),7,'u',{'model_load_start':10,'model_load_end':20,'request_start':19,'request_end':30,'teardown_end':40},15)
  with self.assertRaises(ValueError):nvml_stages.summarize(self.samples()[1:],7,'u',{'model_load_start':10,'model_load_end':20,'request_start':21,'request_end':30,'teardown_end':40},15)
 def test_other_compute_rejected(self):
  s=self.samples();s[2]['processes'].append({'pid':8,'type':'C','used_bytes':M})
  with self.assertRaises(ValueError):nvml_stages.summarize(s,7,'u',{'model_load_start':10,'model_load_end':20,'request_start':21,'request_end':30,'teardown_end':40},15)
if __name__=='__main__':unittest.main()
