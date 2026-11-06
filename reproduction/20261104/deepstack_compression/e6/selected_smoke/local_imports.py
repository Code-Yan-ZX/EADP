"""Load this experiment and its critical legacy imports by resolved source path."""
import importlib.util
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
E4=ROOT/'e4'
BASE=HERE.parent/'selected_baselines'

def pinned(name,path):
 path=path.resolve()
 if name in sys.modules:
  module=sys.modules[name]
  if not getattr(module,'__file__',None) or Path(module.__file__).resolve()!=path:
   raise ValueError('conflicting import '+name)
  return module
 spec=importlib.util.spec_from_file_location(name,path)
 if spec is None or spec.loader is None:raise ValueError('missing module '+str(path))
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module
 try:spec.loader.exec_module(module)
 except BaseException:
  del sys.modules[name];raise
 return module

def prepare():
 # E4's verifier intentionally imports its own module called smoke_contract.
 # This experiment's identically named contract is only imported under a
 # separate unique name, and preloaded conflicting modules cause refusal.
 sys.path[:0]=[str(BASE),str(E4)]
 for name,path in (('generation_contract',HERE/'generation_contract.py'),('recover_indices',BASE/'recover_indices.py'),('identity_records',E4/'identity_records.py'),
                   ('decode_observer',E4/'decode_observer.py'),('skip_deepstack',E4/'skip_deepstack.py'),('off_adapter',E4/'off_adapter.py'),
                   ('budget_adapter',E4/'budget_adapter.py'),('adapter',BASE/'adapter.py'),
                   ('completion',ROOT/'e1_analysis/completion.py'),('identity_worker',E4/'identity_worker.py'),
                   ('smoke_contract',E4/'smoke_contract.py'),('verify_k256',E4/'verify_k256.py')):
  pinned(name,path)
 pinned('baseline_runner',BASE/'baseline_runner.py')
 return (pinned('eadp_selected_stock_capture',HERE/'stock_capture.py'),
         pinned('eadp_selected_smoke_contract',HERE/'smoke_contract.py'))
