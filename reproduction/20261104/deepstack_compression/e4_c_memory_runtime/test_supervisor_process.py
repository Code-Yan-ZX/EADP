"""Exercise actual process supervision with a fake sampler; no NVML/CUDA."""
import json,sys,tempfile,time,unittest
from pathlib import Path
from sampler_supervisor import SamplerSupervisor
BASE=Path(__file__).resolve().parent.parent
class SupervisorProcessTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name);self.mon=self.p/'monitor';self.ready=self.p/'ready.json'
  script=self.p/'fake.py';script.write_text('import json,pathlib,sys,time\np=pathlib.Path(sys.argv[1]); p.joinpath("ready.json").write_text(json.dumps({"started":{},"first_sample":{}}))\nwhile not p.joinpath("finish").exists(): time.sleep(.05)\n')
  worker=[sys.executable,str(script),str(self.p)];command=[sys.executable,str(BASE/'monitor/supervise.py'),'--run-dir',str(self.mon),'--timeout','30','--snapshot-seconds','3600','--poll-seconds','1','--grace-seconds','2','--']+worker
  self.s=SamplerSupervisor(command,worker,self.mon,self.ready)
 def test_real_supervisor_ready_and_complete(self):
  self.s.start();self.s.wait_ready(10);(self.p/'finish').touch();self.assertEqual(self.s.wait_finished(10)['reason'],'completed')
 def test_real_supervisor_term_reaps_worker(self):
  self.s.start();self.s.wait_ready(10);self.s.stop(timeout=10)
  terminal=json.loads((self.mon/'terminal.json').read_text());self.assertEqual(terminal['remaining_process_group'],[])
if __name__=='__main__':unittest.main()
