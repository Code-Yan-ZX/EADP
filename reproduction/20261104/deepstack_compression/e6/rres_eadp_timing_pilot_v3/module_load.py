"""Register path-loaded modules so class introspection sees their true origin."""
import importlib.util,sys
from pathlib import Path
def load(name,path):
 path=Path(path).resolve()
 if name in sys.modules:raise ValueError('preloaded module '+name)
 spec=importlib.util.spec_from_file_location(name,path)
 if spec is None or spec.loader is None:raise ValueError('unloadable module')
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module
 try:spec.loader.exec_module(module)
 except BaseException:
  del sys.modules[name]
  raise
 return module
