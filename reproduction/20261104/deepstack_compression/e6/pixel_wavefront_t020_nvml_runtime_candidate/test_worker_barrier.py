import json,tempfile,unittest
from pathlib import Path
from worker_barrier import handshake
class BarrierTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)
  self.e={'mode':'t0_author','nonce':'a'*32,'container_id':'b'*64,'gpu_uuid':'GPU-abcd','host_pid':42,'start_ticks':100,'boot_id':'boot','max_sample_gap_ns':100}
  (self.p/'expected.json').write_text(json.dumps(self.e));self.release={'status':'release','mode':'t0_author','nonce':self.e['nonce'],'container_id':self.e['container_id'],'first_sample_ns':10};(self.p/'release.json').write_text(json.dumps(self.release))
 def test_pre_cuda_release(self):
  e,r=handshake(self.p,self.e['nonce'],lambda:False,namespace_pid=lambda:1);self.assertEqual(e,self.e);self.assertEqual(r,self.release);self.assertFalse(json.loads((self.p/'worker_ready.json').read_text())['cuda_initialized'])
 def test_wrong_nonce_pid_cuda_and_release_fail_closed(self):
  for name in ('nonce','pid','cuda','release'):
   with self.subTest(name=name),tempfile.TemporaryDirectory() as d:
    p=Path(d);(p/'expected.json').write_text(json.dumps(self.e));r=dict(self.release)
    if name=='release':r['container_id']='f'*64
    (p/'release.json').write_text(json.dumps(r))
    with self.assertRaises(ValueError):handshake(p,'c'*32 if name=='nonce' else self.e['nonce'],lambda:name=='cuda',namespace_pid=lambda:2 if name=='pid' else 1)
 def test_reused_ready_rejected(self):
  (self.p/'worker_ready.json').write_text('{}')
  with self.assertRaises(FileExistsError):handshake(self.p,self.e['nonce'],lambda:False,namespace_pid=lambda:1)
 def test_missing_expected_times_out(self):
  (self.p/'expected.json').unlink();clock=iter([0,0,1])
  with self.assertRaises(TimeoutError):handshake(self.p,self.e['nonce'],lambda:False,namespace_pid=lambda:1,clock=lambda:next(clock),sleep=lambda _:None,timeout=1)
if __name__=='__main__':unittest.main()
