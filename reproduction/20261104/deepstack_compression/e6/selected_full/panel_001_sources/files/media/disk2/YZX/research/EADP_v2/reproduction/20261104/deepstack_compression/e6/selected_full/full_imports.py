"""Pin approved smoke dependencies and independent summary-only stock observer."""
import importlib.util,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;SMOKE=HERE.parent/'selected_smoke'
def load(name,path):
 path=path.resolve()
 if name in sys.modules:
  module=sys.modules[name]
  if not getattr(module,'__file__',None) or Path(module.__file__).resolve()!=path:raise ValueError('module source collision: '+name)
  return module
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module
 try:spec.loader.exec_module(module)
 except BaseException:del sys.modules[name];raise
 return module
def prepare():
 helpers=load('eadp_full_smoke_imports',SMOKE/'local_imports.py');_stock,contract=helpers.prepare()
 return load('eadp_full_stock_capture',HERE/'stock_capture.py'),contract
