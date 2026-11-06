import subprocess,sys,tempfile,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
class Locks(unittest.TestCase):
 def test_all_actual_cli_entries_fail_before_output_or_gpu(self):
  with tempfile.TemporaryDirectory() as t:
   cases=[('worker.py',['--job-index','0','--out',t,'--nonce','a'*32]),('sampler_entry.py',['--directory',t]),('controller_entry.py',['--job-index','0']),('launch.py',['--run-id','001'])]
   for file,args in cases:
    p=subprocess.run([sys.executable,str(D/file),*args],capture_output=True,text=True,timeout=10)
    self.assertNotEqual(p.returncode,0);self.assertIn('GPU locked',p.stderr);self.assertEqual(list(Path(t).iterdir()),[])
