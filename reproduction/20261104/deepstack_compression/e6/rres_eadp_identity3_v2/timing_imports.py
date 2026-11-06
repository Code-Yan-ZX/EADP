"""Load frozen timing code alongside E4's already imported, distinct runner."""
import importlib.util,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];TIMING=ROOT/'e6/rres_eadp_timing_pilot_v4'
def load_module(name,path):
 if name in sys.modules:raise ValueError('preloaded timing module '+name)
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module
 try:spec.loader.exec_module(module)
 except BaseException:del sys.modules[name];raise
 return module
def load():
 """The uniquely named timing runner coexists with E4's independent runner."""
 names=('generation_capture','qwen_request','process_core','paired_runner')
 if any(name in sys.modules for name in names):raise ValueError('timing modules unexpectedly preloaded')
 try:
  runner=load_module('paired_runner',TIMING/'paired_runner.py')
  capture=load_module('generation_capture',TIMING/'generation_capture.py')
  request=load_module('qwen_request',TIMING/'qwen_request.py');core=load_module('process_core',TIMING/'process_core.py')
  if request.runner is not runner or core.runner is not runner or core.qwen_request is not request or request.generation_capture is not capture:raise ValueError('incorrect object bindings')
  return request,core,runner
 except BaseException:
  for name in names:sys.modules.pop(name,None)
  raise
