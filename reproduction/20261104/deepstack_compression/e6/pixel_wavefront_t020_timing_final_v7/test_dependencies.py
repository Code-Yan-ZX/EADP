"""Real host-cwd dependency imports, without loading a model or querying GPU."""
import json, os, subprocess, sys, unittest
from pathlib import Path

D=Path(__file__).resolve().parent;B=D.parents[1]
PROGRAM=r'''
import hashlib,importlib.util,json,sys,types
from pathlib import Path
D=Path(sys.argv[1]); B=D.parents[1]; ROOT=B.parents[2]
s=importlib.util.spec_from_file_location('actual_launch',D/'launch.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
paths=[m.R/'controller.py',m.R/'host_runner.py',B/'e6/pixel_wavefront_t020_timing_controller_draft/job_contract.py',m.C/'reference.py',D/'raw_prefix.py',D/'gpu_bounded.py',ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_panel.py']
source={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
if sys.argv[2]=='polluted':
 x=types.ModuleType('_pixel_t020_final_bounded');x.__file__='/tmp/foreign.py';sys.modules[x.__name__]=x
 try:m.load_dependencies(source)
 except ValueError:print('refused')
 else:raise AssertionError('foreign module accepted')
else:
 modules=m.load_dependencies(source);assert len(modules)==7;print(json.dumps([str(Path(v.__file__).resolve()) for v in modules]))
'''
class Dependencies(unittest.TestCase):
 def test_real_host_cwd(self):
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
  p=subprocess.run([sys.executable,'-c',PROGRAM,str(D),'normal'],cwd=B,env=env,capture_output=True,text=True,timeout=60,check=True)
  self.assertEqual(len(json.loads(p.stdout)),7)
 def test_foreign_preload_rejected(self):
  p=subprocess.run([sys.executable,'-c',PROGRAM,str(D),'polluted'],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=60,check=True)
  self.assertIn('refused',p.stdout)
