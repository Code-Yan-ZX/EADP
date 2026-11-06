import subprocess,sys,unittest
from pathlib import Path
import single_arm
class Tests(unittest.TestCase):
 def test_public_entry_and_cli_are_locked(self):
  with self.assertRaisesRegex(RuntimeError,'GPU locked'):single_arm.run_one('x','1','author','unused')
  p=subprocess.run([sys.executable,str(Path(single_arm.__file__))],capture_output=True,text=True,timeout=10)
  self.assertNotEqual(p.returncode,0);self.assertIn('GPU locked',p.stderr)
if __name__=='__main__':unittest.main(verbosity=2)
