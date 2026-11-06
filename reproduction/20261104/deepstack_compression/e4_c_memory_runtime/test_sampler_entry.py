import json,tempfile,unittest
from pathlib import Path
from sampler_entry import launch
class SamplerEntryTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)
  self.e={'mode':'C-ON','nonce':'a'*32,'container_id':'b'*64,'gpu_uuid':'GPU-abcd','host_pid':42,'start_ticks':100,'boot_id':'boot','max_sample_gap_ns':1000000000};(self.p/'expected.json').write_text(json.dumps(self.e))
 def test_paths_and_completed_sample(self):
  seen=[]
  def fake(*a):seen.extend(a);return {'status':'completed','samples':2}
  launch(self.p,self.p/'expected.json',100000000,2000000000,sample=fake)
  self.assertEqual(seen[0],self.e);self.assertEqual(seen[1],self.p/'samples.jsonl');self.assertEqual(json.loads((self.p/'sampler_complete.json').read_text())['samples'],2)
 def test_bad_path_bounds_or_failure_never_completes(self):
  with self.assertRaises(ValueError):launch(self.p,self.p/'elsewhere.json',100000000,2000000000,sample=lambda *a:{})
  with self.assertRaises(ValueError):launch(self.p,self.p/'expected.json',1,2000000000,sample=lambda *a:{})
  with self.assertRaises(ValueError):launch(self.p,self.p/'expected.json',100000000,2000000000,sample=lambda *a:{'status':'failed'})
  self.assertFalse((self.p/'sampler_complete.json').exists())
if __name__=='__main__':unittest.main()
