import json,tempfile,unittest
from pathlib import Path
import job_contract
class Tests(unittest.TestCase):
 def test_persisted_single_job_and_failure(self):
  job={'task':'t','qid':'1','order':0};ref={'processor':{},'selected_indices':[1],'tokens':[2],'eos_ids':[2],'generation_kwargs':{}}
  record=dict(job,processor={},selected_indices=[1],output_ids=[2],eos_ids=[2],generation_scalars={},start_monotonic_ns=10,first_token_monotonic_ns=20,completion_monotonic_ns=30,start_unix_ns=100,completion_unix_ns=120,first_token_synchronized=True,gpu_synchronized_before_start=True)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'raw.json';p.write_text(json.dumps({'job':job,'input_length':1,'full_ids':[5,2]}));x=job_contract.check(job,record,ref,p,d);self.assertEqual(x['ttft_ns'],10);self.assertNotIn('all_samples_retained',record)
   p.write_text(json.dumps({'job':job,'input_length':1,'full_ids':[5,3]}))
   with self.assertRaises(ValueError):job_contract.check(job,record,ref,p,d)
if __name__=='__main__':unittest.main()
