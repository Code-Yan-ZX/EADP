"""Load panel-local modules by fixed absolute path under unique module names."""
import importlib.util
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ALLOWED=frozenset(('panel_gate','native_panel_gate','panel_launch','panel_verify'))

def load_local(name):
    if name not in ALLOWED:raise ValueError('unexpected local module')
    path=(HERE/(name+'.py')).resolve();key='eadp_panel_v2_'+name
    if key in sys.modules:
        module=sys.modules[key]
        if Path(module.__file__).resolve()!=path:raise ValueError('local module identity mismatch')
        return module
    spec=importlib.util.spec_from_file_location(key,path)
    module=importlib.util.module_from_spec(spec);sys.modules[key]=module
    try:spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[key]
        raise
    return module
