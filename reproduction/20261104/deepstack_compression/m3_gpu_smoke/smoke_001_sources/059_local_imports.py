"""Pin new smoke modules and frozen teacher dependencies to exact files."""
import importlib.util
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ORACLE=HERE.parent/'m3_oracle'

def _load(name,path):
    path=path.resolve()
    if name in sys.modules:
        module=sys.modules[name]
        if Path(module.__file__).resolve()!=path:raise ValueError('conflicting module source: '+name)
        return module
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    try:spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module

def local(name):
    if name not in ('contract','core','gate','launcher'):raise ValueError('unknown smoke module')
    return _load('eadp_m3_gpu_'+name,HERE/(name+'.py'))

def teacher():
    _load('teacher_contract',ORACLE/'teacher_contract.py')
    return _load('eadp_m3_oracle_teacher',ORACLE/'oracle_teacher.py'),_load('eadp_m3_saliency',ORACLE/'saliency.py')
