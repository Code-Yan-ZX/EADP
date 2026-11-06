"""Fake-only tests exercise the exact v6 worker main around the GPU gate."""
import importlib.util,json,os,shutil,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
B=Path(__file__).resolve().parents[2];E=B/'e6/pixel_wavefront_t020_nvml_execution_candidate_v8'
spec=importlib.util.spec_from_file_location('_pixel_nvml_worker_main_test',E/'worker.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
class WorkerMainTest(unittest.TestCase):
 def run_case(self,fail_after=False,invalid=False):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);shutil.copyfile(E/'gate.py',d/'gate.py');out=d/'run_001/job_000';out.mkdir(parents=True)
   job={'order':0,'task':'fake','qid':'1','arm':'t0_author'};approval={'jobs':[job]*9,'source_sha256':{}};calls=[]
   def gate():
    calls.append('gate')
    if fail_after and calls.count('gate')==2:raise ValueError('post source changed')
    return approval
   fake_gate=types.SimpleNamespace(admit_gpu=gate);inputmod=types.SimpleNamespace(load_accepted_e1_references=lambda *args:([{'task':'fake','qid':'1'}],{('fake','1'):{'fake':1}}));prepared={'torch':'fake'}
   args=['worker.py','--job-index','0','--out',str(out if not invalid else d/'wrong'),'--nonce','n']
   refs={('fake','1','t0_author'):{'fake':'ref'}}
   with patch.object(worker,'D',d),patch.object(worker,'load',return_value=fake_gate),patch.object(worker,'prepare_dependencies',return_value=(refs,prepared,inputmod,'barrier','model')),patch.object(worker,'run_admitted',side_effect=lambda *a:calls.append('generate')),patch.object(sys,'argv',args):
    if fail_after or invalid:
     with self.assertRaises(ValueError):worker.main()
    else:self.assertEqual(worker.main(),0)
   return calls,(out/'worker_complete.json').exists()
 def test_main_rechecks_before_complete(self):
  calls,done=self.run_case();self.assertTrue(done);self.assertEqual(calls,['gate','generate','gate'])
  calls,done=self.run_case(fail_after=True);self.assertFalse(done);self.assertEqual(calls,['gate','generate','gate'])
 def test_wrong_output_never_starts_model(self):
  calls,done=self.run_case(invalid=True);self.assertFalse(done);self.assertEqual(calls,['gate'])
if __name__=='__main__':unittest.main()
