import json,subprocess,tempfile,unittest
from pathlib import Path
from sampler_supervisor import SamplerSupervisor
class P:
 def __init__(self,*a,**k):self.rc=None;self.terminated=False
 def poll(self):return self.rc
 def wait(self,timeout):
  if self.rc is None:raise subprocess.TimeoutExpired('fake',timeout)
  return self.rc
 def terminate(self):self.terminated=True;self.rc=1
 def kill(self):self.rc=-9
class SupervisorTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name);self.mon=self.p/'monitor';self.ready=self.p/'ready.json';self.s=SamplerSupervisor(['monitor','--','sampler'],['sampler'],self.mon,self.ready,popen=P)
 def artifacts(self):
  self.mon.mkdir();(self.mon/'started.json').write_text(json.dumps({'worker_pid':42,'process_group':42,'command':['sampler']}));self.ready.write_text(json.dumps({'started':{},'first_sample':{}}))
 def test_ready_and_real_terminal_fields(self):
  self.s.start();self.artifacts();self.assertIn('first_sample',self.s.wait_ready(1));self.s.process.rc=0
  (self.mon/'terminal.json').write_text(json.dumps({'worker_pid':42,'worker_returncode':0,'reason':'completed','remaining_process_group':[]}));self.assertEqual(self.s.wait_finished(1)['worker_returncode'],0)
 def test_wrong_worker_command_rejected_before_release(self):
  self.s.start();self.artifacts();p=self.mon/'started.json';v=json.loads(p.read_text());v['command']=['impostor'];p.write_text(json.dumps(v))
  with self.assertRaisesRegex(ValueError,'identity'):self.s.wait_ready(1)
 def test_exit_before_ready_or_reused_paths_rejected(self):
  self.s.start();self.s.process.rc=1
  with self.assertRaises(ValueError):self.s.wait_ready(1)
  x=SamplerSupervisor(['monitor','--','sampler'],['sampler'],self.mon,self.ready,popen=P);self.mon.mkdir()
  with self.assertRaises(ValueError):x.start()
 def test_terminal_with_orphan_or_failure_rejected(self):
  self.s.start();self.artifacts();self.s.process.rc=0
  (self.mon/'terminal.json').write_text(json.dumps({'worker_pid':42,'worker_returncode':0,'reason':'completed','remaining_process_group':[43]}))
  with self.assertRaises(ValueError):self.s.wait_finished(1)
 def test_stop_missing_terminal_is_unknown(self):
  self.s.start();self.artifacts()
  with self.assertRaisesRegex(RuntimeError,'terminal unavailable'):self.s.stop()
 def test_stop_requests_supervisor_termination(self):
  self.s.start();self.artifacts();(self.mon/'terminal.json').write_text(json.dumps({'worker_pid':42,'worker_returncode':-15,'reason':'interrupted','remaining_process_group':[]}));self.s.stop();self.assertTrue(self.s.process.terminated)
if __name__=='__main__':unittest.main()
