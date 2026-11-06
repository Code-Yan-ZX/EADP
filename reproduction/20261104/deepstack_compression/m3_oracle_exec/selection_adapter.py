"""Diagnostics-only custom K256 P3/DS111 selection; no model initialization.

The original author greedy still runs and is checked against the frozen
baseline. Only the pruner output is replaced, then E4's actual prefill/DS
checks run unchanged. This module is not an authorization to use the GPU.
"""
import importlib.util
from pathlib import Path
import sys

E4=Path(__file__).resolve().parent.parent/'e4'
sys.path.insert(1,str(E4))
for name in ('off_adapter','skip_deepstack'):
    if name in sys.modules and Path(sys.modules[name].__file__).resolve()!=(E4/(name+'.py')).resolve():
        raise ValueError('conflicting E4 module: '+name)
spec=importlib.util.spec_from_file_location('m3_oracle_exec_budget_adapter',E4/'budget_adapter.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class SelectedAdapter(mod.BudgetAdapter):
    def __init__(self,torch,wrapper,pruner_module,spatial,expected_greedy,expected_full=None):
        if (len(spatial)!=256 or spatial!=sorted(spatial) or len(set(spatial))!=256 or
            any(type(i)!=int or i<0 or i>=1024 for i in spatial) or
            len(expected_greedy)!=256 or len(set(expected_greedy))!=256 or
            any(type(i)!=int or i<0 or i>=1024 for i in expected_greedy)):
            raise ValueError('invalid custom or baseline indices')
        super().__init__(torch,wrapper,pruner_module,256,True)
        self.spatial_ids=list(spatial);self.expected_greedy=list(expected_greedy);self.expected_full=expected_full
    def _replace(self,_module,_args,result):
        s=self.state;t=self.torch
        if (s.get('visual_calls')!=1 or s.get('greedy_calls')!=1 or not isinstance(result,(tuple,list)) or
            len(result)!=2 or result[1]!=[256]):raise ValueError('unexpected author pruner path')
        order=s['ordered'].detach().cpu().tolist()
        if order!=self.expected_greedy:raise ValueError('actual greedy differs from frozen baseline')
        original=t.tensor(sorted(order),device=s['visual'].device,dtype=t.long)
        if not t.equal(result[0],s['visual'].index_select(0,original)):raise ValueError('original pruner feature mismatch')
        index=t.tensor(self.spatial_ids,device=s['visual'].device,dtype=t.long)
        chosen=s['visual'].index_select(0,index)
        s.update(author_greedy=order,author_pruned=result[0].detach().clone(),ordered=index,pruned=chosen,splits=result[1],
                 custom_selection=True)
        return chosen,result[1]
    def __enter__(self):
        super().__enter__()
        try:
            handle=self.w.pruner.register_forward_hook(self._replace);self.stack.callback(handle.remove)
            return self
        except BaseException:
            self.stack.close();self.stack=None;raise
    def _prefill(self,module,args,kw):
        out=super()._prefill(module,args,kw)
        if out is not None:
            full=self.expected_full
            if full is not None:
                s=self.state
                if any(not self.torch.equal(s['inputs'][k],v) for k,v in full['processor'].items()):raise ValueError('processor changed during generation')
                if not self.torch.equal(s['visual'],full['streams'][0]) or any(not self.torch.equal(a,b) for a,b in zip(s['deepstack'],full['streams'][1:])):
                    raise ValueError('full visual streams changed during generation')
            self.state['selection_mode']='oracle_fixed_indices'
        return out
    def verify_calls(self):
        return super().verify_calls() and self.state.get('custom_selection') is True
