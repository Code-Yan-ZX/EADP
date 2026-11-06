import os,subprocess,sys,unittest
from pathlib import Path
D=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def execute(self,code):return subprocess.run([sys.executable,'-c',code],cwd=D,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=120)
 def test_actual_cpu_import_and_six_references(self):
  r=self.execute("import model_candidate; x=model_candidate.prepare_cpu(); assert len(x['references'])==6 and not x['torch'].cuda.is_initialized()")
  self.assertEqual(r.returncode,0,r.stdout+r.stderr)
 def test_foreign_model_source_rejected(self):
  code="import sys,types; x=types.ModuleType('pixelprune');x.__file__='/tmp/foreign.py';sys.modules['pixelprune']=x;import model_candidate\ntry:model_candidate.prepare_cpu()\nexcept ValueError as e:assert 'foreign' in str(e)\nelse:raise AssertionError('foreign module accepted')"
  r=self.execute(code);self.assertEqual(r.returncode,0,r.stdout+r.stderr)
if __name__=='__main__':unittest.main()
