"""Diagnostic model hook for E2/E3; requires real-model endpoint validation.

This adapter only changes the three selected DS streams supplied at prefill.
It does not avoid any visual computation and is not a performance runner.
"""
from pathlib import Path
import sys

E4 = Path(__file__).resolve().parent.parent / 'e4'
sys.path.insert(0, str(E4))
import budget_adapter as _budget_module
import interventions as _intervention_module
if Path(_budget_module.__file__).resolve() != (E4 / 'budget_adapter.py').resolve():
    raise ImportError('unexpected budget_adapter module')
for _name in ('off_adapter', 'skip_deepstack'):
    _module = sys.modules.get(_name)
    if _module is None or Path(_module.__file__).resolve() != (E4 / (_name + '.py')).resolve():
        raise ImportError('unexpected ' + _name + ' module')
if Path(_intervention_module.__file__).resolve() != Path(__file__).resolve().with_name('interventions.py'):
    raise ImportError('unexpected interventions module')
from budget_adapter import BudgetAdapter
from interventions import e2_slots, e3_streams


class InterventionAdapter(BudgetAdapter):
    def __init__(self, torch, wrapper, pruner_module, family, arm, image_sha256):
        super().__init__(torch, wrapper, pruner_module, 256, True)
        if family == 'E2' and arm not in ('000', '100', '010', '001', '110', '101', '011', '111'):
            raise ValueError('invalid E2 arm')
        if family == 'E3' and arm not in ('OFF', 'MATCH', 'PERM', 'MEAN'):
            raise ValueError('invalid E3 arm')
        if family not in ('E2', 'E3'):
            raise ValueError('invalid intervention family')
        self.family, self.arm, self.image_sha256 = family, arm, image_sha256

    def _prefill(self, module, args, kwargs):
        result = super()._prefill(module, args, kwargs)
        if result is None:
            return None
        args, kwargs = result
        original = self.state['selected_deepstack']
        if self.family == 'E2':
            streams = e2_slots(self.torch, original, self.arm)
            metadata = {'family': self.family, 'arm': self.arm}
        else:
            streams, metadata = e3_streams(self.torch, original, self.arm, self.image_sha256)
            metadata = dict(metadata, family=self.family, arm=self.arm)
        self.state['intervention_metadata'] = metadata
        self.state['selected_before_intervention'] = [z.detach().clone() for z in original]
        # Parent observer checks these values against the actual LLM hook, then
        # snapshots them before the language model can mutate the arguments.
        self.state['selected_deepstack'] = streams
        kwargs['deepstack_visual_embeds'] = streams
        return args, kwargs
