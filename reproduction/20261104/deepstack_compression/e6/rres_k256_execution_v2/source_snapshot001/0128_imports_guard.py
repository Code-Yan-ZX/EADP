"""Reject preloaded/lookalike modules before they can be called."""
import sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];L=B/'e4_c_latency'
LATENCY={name:L/(name+'.py') for name in ('identity_gate','input_preflight','contract','timer','runner','real_worker','timing_launch')}
def check(mapping,require=False):
 for name,path in mapping.items():
  module=sys.modules.get(name)
  if module is None:
   if require:raise ValueError('missing imported module '+name)
  elif not getattr(module,'__file__',None) or Path(module.__file__).resolve()!=Path(path).resolve():raise ValueError('foreign module '+name)
