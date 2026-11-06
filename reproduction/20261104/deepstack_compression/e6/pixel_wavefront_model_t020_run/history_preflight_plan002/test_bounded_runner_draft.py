import subprocess,sys,unittest
from pathlib import Path
import bounded_runner_draft as r
class Tests(unittest.TestCase):
 def test_locked_before_artifact_creation(self):
  out=Path(__file__).parent/'never_created_test_output'
  with self.assertRaisesRegex(RuntimeError,'GPU locked'):r.run_arm({'arm':'author','order':0},out)
  self.assertFalse(out.exists())
 def test_worker_entry_locked(self):
  p=subprocess.run([sys.executable,str(Path(__file__).parent/'worker_entry.py')],text=True,capture_output=True,timeout=10)
  self.assertNotEqual(p.returncode,0);self.assertIn('GPU locked',p.stderr)
if __name__=='__main__':unittest.main(verbosity=2)
