"""Reject preloaded same-name modules from outside frozen source paths."""
import importlib,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent

def check_existing(bindings):
 for name,path in bindings.items():
  item=sys.modules.get(name)
  if item is not None and (not getattr(item,'__file__',None) or Path(item.__file__).resolve()!=Path(path).resolve()):raise ValueError('foreign cached module: '+name)

def checked_import(name,path):
 check_existing({name:path});module=importlib.import_module(name)
 if not getattr(module,'__file__',None) or Path(module.__file__).resolve()!=Path(path).resolve():raise ValueError('wrong module source: '+name)
 return module

def latency_bindings():
 p=B/'e4_c_latency';return {name:p/(name+'.py') for name in ('identity_gate','latency_gate','latency_worker','real_worker','input_preflight','schedule','runner','timer','stage_runner','contract','timing_launch','light_adapter')}
