"""Exercise the real CLI with an audit hook that traps forbidden file reads."""
import os,subprocess,sys,unittest
from pathlib import Path
import score

class RealCliTests(unittest.TestCase):
 def test_unapproved_cli_never_opens_target_or_truth(self):
  root=score.B;script=score.D/'score.py';out=score.D/'score001'
  if out.exists():raise ValueError('test expects unused output')
  code="""import os,runpy,sys\nfrom pathlib import Path\nforbidden=[str(Path(x).resolve()) for x in sys.argv[3:]]\ndef audit(event,args):\n if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):\n  p=str(Path(os.fsdecode(args[0])).resolve())\n  if any(p==x or p.startswith(x+os.sep) for x in forbidden): raise RuntimeError('TRUTH_OR_TARGET_SENTINEL')\nsys.addaudithook(audit)\nsys.argv=[sys.argv[1],'--out',sys.argv[2]]\nrunpy.run_path(sys.argv[0],run_name='__main__')\n"""
  banned=[root/'e6/rres_k256_dev384/run001/observations',Path('/media/disk2/YZX/LMUData/AI2D_TEST.tsv'),Path('/media/disk2/YZX/LMUData/DocVQA_VAL.tsv'),Path('/media/disk2/YZX/LMUData/TextVQA_VAL.tsv')]
  p=subprocess.run([sys.executable,'-c',code,str(script),str(out),*[str(x) for x in banned]],cwd=root,capture_output=True,text=True,timeout=120)
  self.assertNotEqual(p.returncode,0);self.assertNotIn('TRUTH_OR_TARGET_SENTINEL',p.stderr);self.assertFalse(out.exists())
if __name__=='__main__':unittest.main()
