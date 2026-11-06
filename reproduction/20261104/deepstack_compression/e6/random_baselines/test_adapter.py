import sys,unittest
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1]
sys.path.insert(0,str(BASE/'e4'))
from test_off_adapter import FakeWrapper
from adapter import RandomAdapter

class AdapterTests(unittest.TestCase):
 def test_real_hook_chain_aligns_selected_main_position_and_ds(self):
  w=FakeWrapper()
  class Language(torch.nn.Module):
   def forward(self,inputs_embeds,attention_mask,position_ids=None,visual_pos_masks=None,deepstack_visual_embeds=None):
    self.ds=deepstack_visual_embeds;self.positions=position_ids;return inputs_embeds.mean(dim=1)
  w.model.model.language_model=Language();original=list(range(255,-1,-1))
  with RandomAdapter(torch,w,w.namespace,'01'*32,20261104,original) as a:
   inp=w._processor_inputs([]);main,_=w.model.visual(inp['pixel_values']);pruned,_=w.pruner(main);ids=a.random_selection['spatial_ids']
   self.assertTrue(torch.equal(pruned,main[ids]))
   embeds=w.model.get_input_embeddings()(inp['input_ids']);embeds=torch.cat((embeds[:,:2],pruned[None],embeds[:,1026:]),dim=1)
   w.model.model.language_model(inputs_embeds=embeds,attention_mask=torch.ones((1,261),dtype=torch.long))
   self.assertTrue(a.verify_calls());self.assertEqual(a.state['selection_mode'],'e6_random_fixed_seed');self.assertEqual(a.state['spatial'].tolist(),ids)
   for actual,full in zip(a.state['actual_deepstack'],a.state['deepstack']):self.assertTrue(torch.equal(actual,full[ids]))
   self.assertEqual(a.state['retained'].tolist(),[0,1]+[2+i for i in ids]+list(range(1026,inp['input_ids'].shape[1])))
  self.assertFalse(w.model.model.language_model._forward_pre_hooks)
 def test_wrong_e1_selection_fails(self):
  w=FakeWrapper()
  with RandomAdapter(torch,w,w.namespace,'01'*32,20261104,list(range(256))) as a:
   x=w._processor_inputs([]);main,_=w.model.visual(x['pixel_values'])
   with self.assertRaisesRegex(ValueError,'baseline'):w.pruner(main)
if __name__=='__main__':unittest.main()
