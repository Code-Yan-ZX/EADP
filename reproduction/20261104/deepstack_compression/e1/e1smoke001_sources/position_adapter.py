"""E1 variants layered on the E0 identity-tested selection adapter."""
from adapter import Adapt


class FourArm(Adapt):
    def __init__(self, torch, wrapper, arm, observer):
        super().__init__(torch, wrapper, 256, 'on' if arm.endswith('D1') else 'off', observer)
        self.arm = arm

    def prefill(self, module, args, kw):
        result = super().prefill(module, args, kw)
        # PyTorch pre-hooks use None to mean unchanged; E0 deliberately
        # returns it on every decode call after its first prefill.
        if result is None: return None
        args, kw = result
        if self.state['llm_calls'] != 1: return args, kw
        kw = dict(kw)
        if self.arm.startswith('P1'):
            mask = kw['attention_mask']
            if not bool(self.torch.all(mask == 1)): raise ValueError('P1 requires no padding')
            kw['position_ids'] = (mask.long().cumsum(-1)-1).unsqueeze(0).expand(3,-1,-1)
            self.w.model.model.rope_deltas = self.torch.zeros((mask.shape[0],1),device=mask.device,dtype=self.torch.long)
        if self.arm.endswith('D0'):
            kw['visual_pos_masks'] = None
            kw['deepstack_visual_embeds'] = None
        return args, kw
