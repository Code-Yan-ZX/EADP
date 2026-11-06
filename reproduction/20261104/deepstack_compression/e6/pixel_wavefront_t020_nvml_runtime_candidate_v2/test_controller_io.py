import json,tempfile,unittest
from pathlib import Path
from controller_io import host_alive,identity_from_inspect,publish,publish_stopped
class IoTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name);(self.p/'sys/kernel/random').mkdir(parents=True);(self.p/'sys/kernel/random/boot_id').write_text('boot');(self.p/'42').mkdir();self.cid='b'*64
  (self.p/'42/stat').write_text('42 (worker) S '+' '.join(['0']*18+['123']+['0']*5));(self.p/'42/cgroup').write_text('0::/docker/'+self.cid)
  self.live={'container_id':self.cid,'running':True,'pid':42};self.e=identity_from_inspect(self.live,'t0_author','a'*32,'GPU-abcd',1000000000,proc=self.p)
 def test_identity_and_exact_stopped_evidence(self):
  raw={'Id':self.cid,'State':{'Running':False,'Pid':0,'ExitCode':0}};x={'container_id':self.cid,'running':False,'pid':0,'exit_code':0,'raw':raw};end=publish_stopped(self.p,self.e,x,lambda _:False,clock=lambda:99)
  self.assertEqual(end,99);self.assertEqual(json.loads((self.p/'stopped.json').read_text())['monotonic_ns'],99)
 def test_wrong_cgroup_and_live_process_rejected(self):
  (self.p/'42/cgroup').write_text('other')
  with self.assertRaises(ValueError):identity_from_inspect(self.live,'t0_author','a'*32,'GPU-abcd',1000000000,proc=self.p)
  with self.assertRaises(ValueError):publish_stopped(self.p,self.e,{'container_id':self.cid,'running':False,'pid':0},lambda _:True)
 def test_host_pid_start_and_boot_reuse_detection(self):
  self.assertTrue(host_alive(self.e,proc=self.p))
  (self.p/'sys/kernel/random/boot_id').write_text('rebooted');self.assertFalse(host_alive(self.e,proc=self.p))
 def test_immutable_publish(self):
  publish(self.p/'x.json',{'a':1})
  with self.assertRaises(FileExistsError):publish(self.p/'x.json',{'a':2})
if __name__=='__main__':unittest.main()
