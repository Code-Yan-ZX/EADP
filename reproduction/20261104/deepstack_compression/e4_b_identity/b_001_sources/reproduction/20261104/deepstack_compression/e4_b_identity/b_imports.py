"""Load B-local modules by absolute source path and unique module identity."""
import importlib.util
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ALLOWED=frozenset(('b_gate','arm','off_capture','pair_contract','verify_records'))
def local(name):
    if name not in ALLOWED:raise ValueError('unexpected B module')
    path=(HERE/(name+'.py')).resolve();key='eadp_b_identity_'+name
    if key in sys.modules:
        module=sys.modules[key]
        if Path(module.__file__).resolve()!=path:raise ValueError('B module identity mismatch')
        return module
    spec=importlib.util.spec_from_file_location(key,path);module=importlib.util.module_from_spec(spec);sys.modules[key]=module
    try:spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[key]
        raise
    return module
