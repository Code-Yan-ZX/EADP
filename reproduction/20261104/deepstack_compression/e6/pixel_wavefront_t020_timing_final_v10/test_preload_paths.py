"""Preloaded modules claiming the true path must not bypass source loading."""
import importlib.util, hashlib, sys, types, unittest
from pathlib import Path
from unittest.mock import patch
import launch, worker, preflight_entry, preflight_launch, preflight_gate
D=Path(__file__).resolve().parent

class PreloadedPath(unittest.TestCase):
 def test_same_path_spoof_rejected(self):
  path=D/'raw_prefix.py';digest=hashlib.sha256(path.read_bytes()).hexdigest()
  for module in (launch,worker,preflight_entry,preflight_launch,preflight_gate):
   name='_same_path_spoof';fake=types.ModuleType(name);fake.__file__=str(path)
   with patch.dict(sys.modules,{name:fake}):
    with self.assertRaises(ValueError):
     if module in (launch,worker,preflight_gate):module.load(name,path,digest)
     else:module.load(name,path)
 def test_preflight_launcher_rejects_same_path_diagnose(self):
  author=preflight_launch.ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02'
  paths=[author/'diagnose.py',author/'run_panel.py'];src={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
  fake=types.ModuleType('diagnose');fake.__file__=str(paths[0])
  with patch.dict(sys.modules,{'diagnose':fake}):
   with self.assertRaises(ValueError):preflight_launch.load_author(src)
