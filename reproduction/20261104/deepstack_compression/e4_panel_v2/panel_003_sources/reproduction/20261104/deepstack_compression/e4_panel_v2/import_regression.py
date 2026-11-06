"""Exercise the worker's real import chain without loading a model or data."""
import json
import os
from pathlib import Path
import runpy
import sys

here = Path(__file__).resolve().parent
worker = runpy.run_path(str(here / 'panel_worker.py'), run_name='panel_import_regression')
load = worker['load_local']
# Exercise the failure mode even if an upstream import has not yet populated it.
sys.path.insert(0, str(here.parent / 'e4'))
import panel_launch as legacy_launch
assert Path(legacy_launch.__file__).resolve() == (here.parent / 'e4/panel_launch.py').resolve()
resolved = {}
for name in ('panel_gate', 'native_panel_gate', 'panel_launch', 'panel_verify'):
    mod = load(name)
    path = Path(mod.__file__).resolve()
    assert path == (here / (name + '.py')).resolve(), (name, path)
    resolved[name] = str(path)
assert worker['validated_dev'] is load('panel_gate').validated_dev
assert worker['check_native'] is load('native_panel_gate').check_native
assert load('panel_verify').load_local('panel_launch') is load('panel_launch')
assert load('panel_launch').check_panel_approval.__module__ == 'eadp_panel_v2_panel_launch'
print(json.dumps({'status':'passed', 'cwd':os.getcwd(), 'pythonpath':os.environ.get('PYTHONPATH'),
                  'legacy_panel_launch':str(Path(legacy_launch.__file__).resolve()),
                  'resolved':resolved, 'gpu_used':False, 'model_loaded':False, 'predictions':False}))
