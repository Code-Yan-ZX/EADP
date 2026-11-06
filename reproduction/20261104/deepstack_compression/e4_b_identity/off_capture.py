"""Identity-only extension for the separately audited C-DROP/C-SKIP path.

Clones actual prefill inputs before the language model can mutate them.  This
instrumentation is diagnostic and must never be used as performance timing.
"""
from pathlib import Path
import sys
E4=Path(__file__).resolve().parent.parent/'e4'
sys.path.insert(1,str(E4))
from off_adapter import E4OffAdapter

class ObservedOffAdapter(E4OffAdapter):
    def _observe_prefill(self,module,args,kw):
        first='actual_ds_off' not in self.state
        super()._observe_prefill(module,args,kw)
        if not first:return
        s=self.state
        s['prefill_embeds']=kw['inputs_embeds'].detach().clone()
        s['prefill_mask']=kw['attention_mask'].detach().clone()
        s['prefill_position_ids']=kw['position_ids'].detach().clone()
        s['adjusted_delta']=self.w.model.model.rope_deltas.detach().clone()
