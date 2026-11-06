"""Fail closed on same-name module contamination before executing dependencies."""
import importlib.util,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];E4=BASE/'e4_c_latency'

def pin(name,path):
 path=Path(path).resolve();existing=sys.modules.get(name)
 if existing is not None:
  if not getattr(existing,'__file__',None) or Path(existing.__file__).resolve()!=path:raise ValueError('foreign preloaded module: '+name)
  return existing
 spec=importlib.util.spec_from_file_location(name,path)
 if spec is None or spec.loader is None:raise ValueError('cannot load pinned module '+name)
 mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
 try:spec.loader.exec_module(mod)
 except BaseException:
  if sys.modules.get(name) is mod:del sys.modules[name]
  raise
 if Path(mod.__file__).resolve()!=path:raise ValueError('module path changed: '+name)
 return mod

def pin_all():
 # real_worker expects the E4 timing contract; the pilot contract is uniquely named.
 for name,path in [('pilot_contract',HERE/'pilot_contract.py'),('observer',HERE/'observer.py'),
                   ('timer',E4/'timer.py'),('identity_gate',E4/'identity_gate.py'),
                   ('input_preflight',E4/'input_preflight.py'),('contract',E4/'contract.py'),
                   ('skip_deepstack',BASE/'e4/skip_deepstack.py'),('light_adapter',E4/'light_adapter.py'),
                   ('runner',E4/'runner.py'),('real_worker',E4/'real_worker.py'),('timing_launch',E4/'timing_launch.py')]:
  pin(name,path)
