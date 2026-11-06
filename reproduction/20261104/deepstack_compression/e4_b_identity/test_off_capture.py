import unittest
from types import SimpleNamespace
import torch
from off_capture import ObservedOffAdapter

class CaptureTests(unittest.TestCase):
    def test_prefill_snapshot_precedes_inplace_mutation(self):
        w=SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(rope_deltas=torch.tensor([[7]]))))
        a=ObservedOffAdapter(torch,w,None,'C-DROP')
        p=torch.tensor([[[1,2]]]);a.state['prefill_position_ids']=p
        kw={'inputs_embeds':torch.tensor([[[3.,4.]]]),'attention_mask':torch.tensor([[1,1]]),
            'position_ids':p,'visual_pos_masks':None,'deepstack_visual_embeds':None}
        a._observe_prefill(None,(),kw)
        kw['inputs_embeds'].zero_();kw['attention_mask'].zero_();p.zero_();w.model.model.rope_deltas.zero_()
        self.assertEqual(a.state['prefill_embeds'].tolist(),[[[3.,4.]]])
        self.assertEqual(a.state['prefill_mask'].tolist(),[[1,1]])
        self.assertEqual(a.state['prefill_position_ids'].tolist(),[[[1,2]]])
        self.assertEqual(a.state['adjusted_delta'].tolist(),[[7]])
    def test_actual_injection_rejected(self):
        w=SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(rope_deltas=torch.tensor([[0]]))))
        a=ObservedOffAdapter(torch,w,None,'C-SKIP');p=torch.tensor([[[1]]]);a.state['prefill_position_ids']=p
        kw={'inputs_embeds':torch.ones(1,1,1),'attention_mask':torch.ones(1,1),'position_ids':p,
            'visual_pos_masks':torch.ones(1,1),'deepstack_visual_embeds':None}
        with self.assertRaises(ValueError):a._observe_prefill(None,(),kw)

if __name__=='__main__':unittest.main()
