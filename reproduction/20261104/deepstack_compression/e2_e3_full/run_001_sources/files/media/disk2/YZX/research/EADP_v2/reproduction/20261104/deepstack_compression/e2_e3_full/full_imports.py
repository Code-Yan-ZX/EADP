"""Pin full-panel local modules without polluting common module names."""
import importlib.util,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def local(name):
 if name not in ('contract','panel_gate','panel_runner','panel_worker','validate_panel_record','source_inventory'):raise ValueError('invalid full local module')
 unique='eadp_e2e3_full_'+name;path=(HERE/(name+'.py')).resolve()
 if unique in sys.modules:
  module=sys.modules[unique]
  if Path(module.__file__).resolve()!=path:raise ImportError('conflicting full-panel local module')
  return module
 spec=importlib.util.spec_from_file_location(unique,path);module=importlib.util.module_from_spec(spec);sys.modules[unique]=module
 try:spec.loader.exec_module(module)
 except BaseException:
  del sys.modules[unique];raise
 return module
