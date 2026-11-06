import copy,unittest
import nvml_contract as n
class Tests(unittest.TestCase):
 def test_native_metadata_unknown_graphics_and_target(self):
  mem={'total':80*n.MIB,'used':None,'free':30*n.MIB,'reserved':None};a={'status':'completed','gpu_uuid':'g','query_start_ns':1,'query_end_ns':2,'processes':[{'pid':99,'type':'G','used_bytes':None}],'device_memory_bytes':mem};b=dict(a,query_start_ns=3,query_end_ns=4,processes=[{'pid':7,'type':'C+G','used_bytes':20*n.MIB}]);c=dict(a,query_start_ns=5,query_end_ns=6);r=n.validate([a,b,c],7,'g',(2,5),4);self.assertIsNone(r[0]['target_used_bytes']);self.assertEqual(r[1]['target_used_bytes'],20*n.MIB);self.assertEqual(r[0]['background_graphics_pids'],[99])
  x=copy.deepcopy(b);x['processes'].append({'pid':9,'type':'C','used_bytes':n.MIB})
  with self.assertRaises(ValueError):n.validate([a,x,c],7,'g',(2,5),4)
  x=copy.deepcopy(b);x['query_end_ns']=9
  with self.assertRaises(ValueError):n.validate([a,x,c],7,'g',(2,5),4)
  x=copy.deepcopy(b);x['processes'][0]['used_bytes']=1
  with self.assertRaises(ValueError):n.validate([a,x,c],7,'g',(2,5),4)
if __name__=='__main__':unittest.main()
