import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

spec=importlib.util.spec_from_file_location('m3_selected_adapter_test',Path(__file__).resolve().parent/'selection_adapter.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class AdapterTests(unittest.TestCase):
    def test_author_original_and_requested_selection_checked(self):
        order=list(range(255,-1,-1));target=list(range(256,512));v=torch.arange(1024*3).reshape(1024,3)
        a=m.SelectedAdapter(torch,SimpleNamespace(),SimpleNamespace(),target,order)
        a.state.update(visual_calls=1,greedy_calls=1,visual=v,ordered=torch.tensor(order))
        value,splits=a._replace(None,None,(v[sorted(order)],[256]))
        self.assertTrue(torch.equal(value,v[target]));self.assertEqual(splits,[256]);self.assertEqual(a.state['author_greedy'],order)
        a.state['ordered']=torch.tensor(order[:-1]+[256])
        with self.assertRaisesRegex(ValueError,'baseline'):a._replace(None,None,(v[sorted(order)],[256]))
    def test_full_hooks_inject_selected_main_positions_and_three_ds(self):
        from test_off_adapter import FakeWrapper
        w=FakeWrapper()
        class Language(torch.nn.Module):
            def forward(self,inputs_embeds,attention_mask,position_ids=None,visual_pos_masks=None,deepstack_visual_embeds=None):
                self.ds=deepstack_visual_embeds;self.positions=position_ids;return inputs_embeds.mean(dim=1)
        w.model.model.language_model=Language();target=list(range(256,512));original=list(range(255,-1,-1))
        with m.SelectedAdapter(torch,w,w.namespace,target,original) as a:
            inp=w._processor_inputs([]);main,_=w.model.visual(inp['pixel_values']);pruned,_=w.pruner(main)
            self.assertTrue(torch.equal(pruned,main[target]))
            embeds=w.model.get_input_embeddings()(inp['input_ids'])
            embeds=torch.cat((embeds[:,:2],pruned.unsqueeze(0),embeds[:,1026:]),dim=1)
            w.model.model.language_model(inputs_embeds=embeds,attention_mask=torch.ones((1,261),dtype=torch.long))
            self.assertTrue(a.verify_calls());self.assertEqual(a.state['selection_mode'],'oracle_fixed_indices')
            self.assertEqual(a.state['adjusted_delta'].item(),768)
            for actual,full in zip(a.state['actual_deepstack'],a.state['deepstack']):
                self.assertTrue(torch.equal(actual,full[target]))
        self.assertFalse(w.model.model.language_model._forward_pre_hooks)
    def test_malformed_selection_rejected(self):
        with self.assertRaises(ValueError):m.SelectedAdapter(torch,None,None,list(range(255))+[254],list(range(256)))
if __name__=='__main__':unittest.main()
