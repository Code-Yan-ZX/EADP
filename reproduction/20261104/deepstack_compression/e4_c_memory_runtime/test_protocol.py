"""Cross-module CPU simulation; does not run Docker, NVML, torch or a model."""
import unittest
from controller_io import contract
from lifecycle import run
from test_lifecycle import Fake
class ProtocolTest(unittest.TestCase):
 def make_ops(self,contended=False):
  e={'mode':'C-ON','nonce':'a'*32,'container_id':'b'*64,'gpu_uuid':'GPU-abcd','host_pid':42,'start_ticks':123,'boot_id':'boot','max_sample_gap_ns':100}
  ids={k:e[k] for k in ('host_pid','start_ticks','boot_id','gpu_uuid')}
  def sample(a,b,used,others=[]):return {'type':'sample','query_start_ns':a,'query_end_ns':b,'monotonic_ns':b,'used_bytes':used,'other_gpu_processes':others,**ids}
  first=sample(11,12,None,[{'pid':99}] if contended else []);middle=sample(20,21,1048576);last=sample(31,32,None)
  x=Fake();x.create_container=lambda:e['container_id'];x.observe_running=lambda _:e
  x.wait_worker_ready=lambda _:{'mode':e['mode'],'nonce':e['nonce'],'container_id':e['container_id'],'namespace_pid':1,'cuda_initialized':False}
  x.wait_sampler_ready=lambda _:{'started':{**e,'status':'sampling','monotonic_ns':10},'first_sample':first}
  x.publish_stopped=lambda *_:30;x.wait_sampler=lambda _:([first,middle,last],{'reason':'completed','worker_returncode':0,'remaining_process_group':[]})
  return x,e
 def test_existing_contract_end_to_end(self):
  x,e=self.make_ops();r=run(x,contract.admit_release,contract.audit);self.assertEqual(r['status'],'passed');self.assertEqual(r['audit']['nvml_process_sampled_peak_bytes'],1048576)
 def test_contention_prevents_worker_release(self):
  x,e=self.make_ops(True);r=run(x,contract.admit_release,contract.audit);self.assertEqual(r['status'],'failed');self.assertNotIn('release',x.calls);self.assertTrue(r['cleanup']['stopped_verified'])
if __name__=='__main__':unittest.main()
