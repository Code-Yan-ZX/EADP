import subprocess,sys,unittest
from pathlib import Path
from unittest.mock import patch
import run
class Tests(unittest.TestCase):
 def test_preloaded_core_is_rejected(self):
  cmd=[sys.executable,'-c',"import runpy,sys,types;sys.modules['core']=types.ModuleType('core');runpy.run_path('run.py')"]
  p=subprocess.run(cmd,cwd=Path(__file__).parent,text=True,capture_output=True,timeout=20)
  self.assertNotEqual(p.returncode,0);self.assertIn('preloaded or changed core',p.stderr)
 def test_prelaunch_or_review_hash_corruption_rejected(self):
  source={str(i):'h' for i in range(6)};pre={'status':'passed','independent_review':True,'reviewer_sha256':'h','source_sha256':source};ind={'reviewer_sha256':'h'};doc={'reviewer_sha256':'h'}
  with patch.object(run,'sha',return_value='h'):
   run.check_reviewers(pre,ind,doc)
   for p,i,d in [({**pre,'status':'failed'},ind,doc),(pre,{'reviewer_sha256':'bad'},doc),(pre,ind,{'reviewer_sha256':'bad'}),({**pre,'source_sha256':{'a':'bad'}},ind,doc)]:
    with self.assertRaises(ValueError):run.check_reviewers(p,i,d)
if __name__=='__main__':unittest.main(verbosity=2)
