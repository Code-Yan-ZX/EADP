import hashlib,json,tempfile,unittest
from pathlib import Path
import test_contract as fixtures
from host_evidence import process_alive,stopped_ns

class HostEvidenceTests(unittest.TestCase):
 def setUp(self):
  self.f=fixtures.MemoryHandshakeTests();self.f.setUp();self.e=self.f.e;self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.d=Path(self.t.name)
 def test_pid_boot_ticks_cgroup_and_zombie(self):
  e=self.e;boot=self.d/'sys/kernel/random/boot_id';boot.parent.mkdir(parents=True);boot.write_text(e['boot_id']);pid=self.d/str(e['host_pid']);pid.mkdir();stat=pid/'stat';stat.write_text('123 (fake process) S '+' '.join(['0']*18+[str(e['start_ticks'])])+'\n');(pid/'cgroup').write_text('0::/docker/'+e['container_id']+'\n')
  self.assertTrue(process_alive(e,self.d));stat.write_text('123 (fake process) Z '+' '.join(['0']*18+[str(e['start_ticks'])]));self.assertFalse(process_alive(e,self.d))
  stat.write_text('123 (fake process) S '+' '.join(['0']*18+['457']));self.assertFalse(process_alive(e,self.d))
 def test_raw_inspect_and_event_identity(self):
  e=self.e;path=self.d/'inspect.json';path.write_text(json.dumps([{'Id':e['container_id'],'State':{'Running':False,'Pid':0}}]));event={k:e[k] for k in ('nonce','container_id','host_pid','start_ticks','boot_id')};event.update(status='stopped',monotonic_ns=20,inspect_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
  self.assertEqual(stopped_ns(e,event,path),20);path.write_text('{}')
  with self.assertRaisesRegex(ValueError,'hash mismatch'):stopped_ns(e,event,path)
if __name__=='__main__':unittest.main()
