"""Exercise the real worker continuation with fake CPU components only."""
import json,os,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import worker
class FakeCuda:
 def is_initialized(self):return False
 def empty_cache(self):pass
 def synchronize(self):pass
class Flow(unittest.TestCase):
 def test_all_arms_pre_cuda_barrier_then_persist_raw_and_teardown(self):
  cid='b'*64;torch=types.SimpleNamespace(cuda=FakeCuda());ref={'processor':{'input_ids':{'shape':[1,2]}}};calls=[]
  for mode in ('t0_author','t020_author','t020_wavefront'):
   with tempfile.TemporaryDirectory() as t:
    out=Path(t);job={'arm':mode};barrier=types.SimpleNamespace(handshake=lambda *a,**k:({'mode':mode,'container_id':cid},{'first_sample_ns':1}))
    def execute(prepared,job,row,e1ref,reference,event,emit):
     calls.append(job['arm']);event({'event':'model_load_start','monotonic_ns':2});emit([[7,8,9]]);event({'event':'request_end','monotonic_ns':3});return object(),object(),{'status':'passed'}
    with patch.object(worker.os,'getpid',return_value=1),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}),patch.object(worker.Path,'read_text',return_value='0::/docker/'+cid):
     value=worker.run_admitted(job,out,'a'*32,{'torch':torch},{},{},ref,barrier,types.SimpleNamespace(execute=execute),clock=lambda:4)
    self.assertEqual(value['status'],'passed');raw=json.loads((out/'raw.json').read_text());self.assertEqual(raw['full_sequence_ids'],[7,8,9]);self.assertEqual(raw['accepted_input_length'],2);self.assertTrue((out/'identity.json').exists());self.assertFalse((out/'worker_complete.json').exists())
  self.assertEqual(calls,['t0_author','t020_author','t020_wavefront'])
 def test_mismatch_at_barrier_never_calls_model(self):
  cid='b'*64;barrier=types.SimpleNamespace(handshake=lambda *a,**k:({'mode':'t0_author','container_id':cid},{'first_sample_ns':1}));fake=types.SimpleNamespace(execute=lambda *a: self.fail('model called'))
  with tempfile.TemporaryDirectory() as t,patch.object(worker.os,'getpid',return_value=1),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}),patch.object(worker.Path,'read_text',return_value='0::/docker/'+cid):
   with self.assertRaises(ValueError):worker.run_admitted({'arm':'t020_author'},Path(t),'a'*32,{'torch':types.SimpleNamespace(cuda=FakeCuda())},{},{},{},barrier,fake)
 def test_post_generation_failure_keeps_raw_and_failure(self):
  cid='b'*64;barrier=types.SimpleNamespace(handshake=lambda *a,**k:({'mode':'t0_author','container_id':cid},{'first_sample_ns':1}));ref={'processor':{'input_ids':{'shape':[1,2]}}}
  def fail(prepared,job,row,e1ref,reference,event,emit):emit([[7,8,9]]);raise ValueError('post-generation failure')
  with tempfile.TemporaryDirectory() as t,patch.object(worker.os,'getpid',return_value=1),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}),patch.object(worker.Path,'read_text',return_value='0::/docker/'+cid):
   out=Path(t)
   with self.assertRaises(ValueError):worker.run_admitted({'arm':'t0_author'},out,'a'*32,{'torch':types.SimpleNamespace(cuda=FakeCuda())},{},{},ref,barrier,types.SimpleNamespace(execute=fail),clock=lambda:4)
   self.assertTrue((out/'raw.json').exists());self.assertTrue((out/'worker_failure.json').exists());self.assertFalse((out/'worker_complete.json').exists())
