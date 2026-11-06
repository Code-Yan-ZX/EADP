import sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import snapshot,verify
class Tests(unittest.TestCase):
 def test_snapshot_rejects_incomplete_run_before_creating_output(self):
  import json
  with tempfile.TemporaryDirectory() as t:
   run=Path(t)/'run';(run/'monitor_run001').mkdir(parents=True);(run/'monitor_run001/started.json').write_text(json.dumps({'worker_pid':1}));(run/'monitor_run001/terminal.json').write_text(json.dumps({'worker_pid':1,'reason':'worker_failed','worker_returncode':1,'remaining_process_group':[]}));(run/'gpu_plan001.json').write_text(json.dumps({'source_sha256':{},'run_supervisor_command':['x'],'run_supervisor_cwd':str(run)}));out=run/'run_001_sources'
   with patch.object(snapshot,'RUN',run),patch.object(sys,'argv',['snapshot.py','--out',str(out)]):
    with self.assertRaises(ValueError):snapshot.main()
   self.assertFalse(out.exists())
 def test_verifier_requires_formal_plan_before_results(self):
  with tempfile.TemporaryDirectory() as t:
   d=Path(t);out=d/'verification001.json'
   with patch.object(verify,'D',d),patch.object(sys,'argv',['verify.py','--out',str(out)]):
    with self.assertRaises(FileNotFoundError):verify.main()
   self.assertFalse(out.exists())
if __name__=='__main__':unittest.main()
