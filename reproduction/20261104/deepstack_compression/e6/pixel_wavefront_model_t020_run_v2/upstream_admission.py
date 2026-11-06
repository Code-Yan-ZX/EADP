"""Load accepted upstream gate without confusing same-named inventories."""
import importlib.util,sys
from pathlib import Path
D=Path(__file__).resolve().parent;P=D.parents[1]/'e6/pixel_wavefront_model_t020'
def load_prior():
 expected=(P/'source_inventory.py').resolve();prior=sys.modules.get('source_inventory')
 if prior is not None and (not getattr(prior,'__file__',None) or Path(prior.__file__).resolve()!=expected):raise ValueError('foreign upstream inventory')
 name='_pixel_t020_original_preflight_gate';path=(P/'admission.py').resolve();mod=sys.modules.get(name)
 if mod is not None:
  if not getattr(mod,'__file__',None) or Path(mod.__file__).resolve()!=path:raise ValueError('foreign upstream gate')
  return mod
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;sys.path.insert(0,str(P))
 try:spec.loader.exec_module(mod)
 except BaseException:del sys.modules[name];raise
 finally:sys.path.remove(str(P))
 inv=sys.modules.get('source_inventory')
 if inv is None or Path(inv.__file__).resolve()!=expected:raise ValueError('upstream inventory path')
 return mod
