"""CPU regression for E0 prehook's None return on decoder calls."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'e0_identity'))
from adapter import Adapt
from position_adapter import FourArm


class PrehookTest(unittest.TestCase):
    def test_prefill_then_decode_none_does_not_break(self):
        wrapper=SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(rope_deltas=None)))
        obj=FourArm(torch,wrapper,'P1-D0',observer=None)
        def parent(fake, module, args, kw):
            fake.state['llm_calls']+=1
            return (args,kw) if fake.state['llm_calls']==1 else None
        with patch.object(Adapt,'prefill',parent):
            args,kw=obj.prefill(None,(),{'attention_mask':torch.ones((1,4),dtype=torch.long)})
            self.assertEqual(kw['position_ids'].tolist(),[[[0,1,2,3]]]*3)
            self.assertEqual(wrapper.model.model.rope_deltas.tolist(),[[0]])
            self.assertIsNone(kw['visual_pos_masks'])
            self.assertIsNone(kw['deepstack_visual_embeds'])
            self.assertIsNone(obj.prefill(None,(),{'attention_mask':torch.ones((1,5),dtype=torch.long)}))

    def test_actual_parent_prefill_then_two_decode_calls(self):
        embedding=torch.nn.Embedding(8,2)
        input_ids=torch.ones((1,1026),dtype=torch.long)
        attention=torch.ones_like(input_ids)
        visual=torch.arange(2048,dtype=torch.float32).reshape(1024,2)
        kept=torch.arange(256)
        observer=SimpleNamespace(errors=[],calls=1,results=[{'ordered_ids':kept,'quotas':torch.tensor([256])}])
        def rope(ids,grid,video,attention_mask):
            return torch.arange(ids.shape[-1]).view(1,1,-1).expand(3,1,-1),torch.tensor([[0]])
        inner=SimpleNamespace(rope_deltas=None,get_rope_index=rope)
        model=SimpleNamespace(model=inner,get_input_embeddings=lambda:embedding)
        obj=FourArm(torch,SimpleNamespace(model=model),'P1-D0',observer)
        obj.state.update(inputs={'input_ids':input_ids,'image_grid_thw':torch.tensor([[1,64,64]]),'attention_mask':attention},
                         start=1,visual=visual,deepstack=[visual.clone() for _ in range(3)],pruned=visual[:256],splits=[256])
        inputs=torch.cat((embedding(input_ids)[:,:1],visual[:256].unsqueeze(0),embedding(input_ids)[:,-1:]),dim=1)
        _,kw=obj.prefill(None,(),{'inputs_embeds':inputs,'attention_mask':torch.ones((1,258),dtype=torch.long)})
        self.assertEqual(kw['position_ids'].shape,(3,1,258))
        self.assertEqual(inner.rope_deltas.tolist(),[[0]])
        for _ in range(2):
            decode={'position_ids':torch.tensor([1])}
            self.assertIsNone(obj.prefill(None,(),decode))
            self.assertEqual(decode['position_ids'].tolist(),[1])

if __name__=='__main__': unittest.main()
