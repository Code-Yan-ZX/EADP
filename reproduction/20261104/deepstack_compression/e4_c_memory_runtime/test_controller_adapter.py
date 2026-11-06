import json,tempfile,unittest
from pathlib import Path
from controller_adapter import ControllerAdapter
class FakeSampler:
 def __init__(self,t):self.t=t
 def wait_finished(self,timeout):return self.t
class AdapterTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)
  command=['monitor','--','sampler'];self.c=ControllerAdapter(None,self.p,'C-ON','a'*32,'GPU-abcd',1000000000,100,command,['sampler'],self.p/'monitor')
 def test_full_sampler_journal_and_ready_binding(self):
  started={'status':'sampling'};sample={'type':'sample','used_bytes':None};summary={'type':'summary','samples':1};(self.p/'samples.jsonl').write_text('\n'.join(json.dumps(x) for x in (started,sample,summary)))
  self.c.ready={'started':started,'first_sample':sample};s,t=self.c.wait_sampler(FakeSampler({'reason':'completed'}));self.assertEqual(s,[sample])
 def test_extra_failure_or_ready_tamper_rejected(self):
  started={'status':'sampling'};sample={'type':'sample'};bad={'type':'failure'};summary={'type':'summary','samples':1};(self.p/'samples.jsonl').write_text('\n'.join(json.dumps(x) for x in (started,sample,bad,summary)))
  self.c.ready={'started':started,'first_sample':sample}
  with self.assertRaisesRegex(ValueError,'journal'):self.c.wait_sampler(FakeSampler({}))
if __name__=='__main__':unittest.main()
