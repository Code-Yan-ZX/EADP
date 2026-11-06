"""Diagnostic adapter for random K256; no launch or model authority.

Reuses the already reviewed arbitrary-selection path. Author greedy is still
computed and verified, so the path cannot provide an unbiased timing result.
"""
import importlib.util,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];M3=BASE/'m3_oracle_exec'
sys.path.insert(0,str(HERE))
from selector import selection
spec=importlib.util.spec_from_file_location('_e6_random_selected_adapter',M3/'selection_adapter.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class RandomAdapter(module.SelectedAdapter):
 def __init__(self,torch,wrapper,pruner_module,image_sha256,seed,expected_e1_greedy,expected_full=None):
  self.random_selection=selection(image_sha256,seed)
  super().__init__(torch,wrapper,pruner_module,self.random_selection['spatial_ids'],expected_e1_greedy,expected_full)
 def _prefill(self,module,args,kwargs):
  result=super()._prefill(module,args,kwargs)
  if result is not None:self.state['selection_mode']='e6_random_fixed_seed'
  return result
