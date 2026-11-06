import importlib.util,json,tempfile,unittest
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parent/'pixel_wavefront_t020_timing_runtime_v3';J=D.parent/'pixel_wavefront_t020_timing_controller_draft'
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
worker=module('synthetic_worker',R/'worker.py');contract=module('synthetic_job_contract',J/'job_contract.py')
class Tests(unittest.TestCase):
 def test_success_preserves_full_ids_before_check(self):
  job={'task':'t','qid':'1','arm':'author'};ref={'processor':{'input_ids':{'shape':[1,2]}},'selected_indices':[4], 'tokens':[8], 'eos_ids':[8], 'generation_kwargs':{'x':1}}
  prepared={'references':{('t','1','author'):ref}}
  def continue_fake(a,j,row,e1,emit):
   emit([[5,6,8]])
   return dict(j,processor_verified=True,processor=ref['processor'],selected_indices=[4],output_ids=[8],eos_ids=[8],generation_scalars={'x':1},start_monotonic_ns=10,first_token_monotonic_ns=20,completion_monotonic_ns=30,start_unix_ns=100,completion_unix_ns=130,first_token_synchronized=True,gpu_synchronized_before_start=True)
  with tempfile.TemporaryDirectory() as t:
   _,m=worker.run_one(job,t,prepared,{}, {},continue_fake,contract.check)
   self.assertEqual(m['ttft_ns'],10);self.assertEqual(json.loads((Path(t)/'raw.json').read_text())['full_ids'],[5,6,8]);self.assertTrue((Path(t)/'record.json').exists())
if __name__=='__main__':unittest.main()
