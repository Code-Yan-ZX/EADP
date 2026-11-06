import copy,unittest
import contract
class Tests(unittest.TestCase):
 def setUp(self):self.anchors=[dict(task=str(i),qid=str(i),group_id=str(i),image_sha256='i',message_sha256='m') for i in range(3)]
 def test_schedule(self):
  x=contract.schedule(self.anchors);self.assertEqual(len(x),24);self.assertEqual(sum(z['phase']=='measurement' for z in x),18);self.assertEqual(x,contract.schedule(self.anchors));self.assertEqual([z['order'] for z in x],list(range(24)))
  for phase in ('warmup','measurement'):
   for rep in range(1 if phase=='warmup' else 3):self.assertEqual(len({(z['task'],z['arm']) for z in x if z['phase']==phase and z['repeat']==rep}),6)
 def test_times_and_identity(self):
  job=contract.schedule(self.anchors)[0];ref=dict(processor={'x':1},selected_indices=[1,2],output_ids=[3],eos_ids=[3],generation_scalars={'a':1});r=dict(job,**ref,start_monotonic_ns=10,first_token_monotonic_ns=20,completion_monotonic_ns=22,start_unix_ns=100,completion_unix_ns=112,first_token_synchronized=True,gpu_synchronized_before_start=True,all_samples_retained=True);self.assertEqual(contract.validate_time(job,r,ref)['ttft_ns'],10)
  for key,bad in [('first_token_monotonic_ns',10),('completion_monotonic_ns',19),('selected_indices',[2]),('first_token_synchronized',False)]:
   x=copy.deepcopy(r);x[key]=bad
   with self.assertRaises(ValueError):contract.validate_time(job,x,ref)
 def test_sampling_missing_and_gaps(self):
  device={'total_mib':80,'used_mib':None,'free_mib':30,'reserved_mib':None};a={'query_start_ns':1,'query_end_ns':3,'gpu_uuid':'G','processes':[],'device':device};b={'query_start_ns':4,'query_end_ns':5,'gpu_uuid':'G','processes':[{'pid':7,'type':'C','used_mib':22}],'device':device};c={'query_start_ns':6,'query_end_ns':7,'gpu_uuid':'G','processes':[],'device':device};r=contract.validate_nvml([a,b,c],7,'G',(2,6),5);self.assertIsNone(r[0]['target_mib']);self.assertEqual(r[1]['target_mib'],22);self.assertFalse(r[-1]['target_present'])
  slow=copy.deepcopy(b);slow['query_end_ns']=10
  with self.assertRaises(ValueError):contract.validate_nvml([a,slow,c],7,'G',(2,6),5)
  with self.assertRaises(ValueError):contract.validate_nvml([b,c],7,'G',(2,6),5)
if __name__=='__main__':unittest.main()
