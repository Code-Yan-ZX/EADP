import types,unittest
import torch
from adapter import SelectedBaselineAdapter
class AdapterTests(unittest.TestCase):
 def make(self):
  seq=1030;start=3;ids=torch.arange(0,1024,4);visual=torch.arange(1024*8,dtype=torch.float32).reshape(1024,8);inputs={'input_ids':torch.arange(seq).reshape(1,-1),'attention_mask':torch.ones((1,seq),dtype=torch.long),'image_grid_thw':torch.tensor([[1,64,64]])};embedding=torch.nn.Embedding(seq,8)
  class Inner:
   def get_rope_index(self,input_ids,image_grid_thw,_video,attention_mask=None):return torch.arange(seq).reshape(1,1,seq).expand(3,-1,-1),torch.tensor([[0]])
  class Model:
   model=Inner()
   def get_input_embeddings(self):return embedding
  pruner=type('DivPruner',(),{})();wrapper=types.SimpleNamespace(pruner=pruner,model=Model());adapter=SelectedBaselineAdapter(torch,wrapper,'div');pruned=visual[ids];original=embedding(inputs['input_ids']);kw={'inputs_embeds':torch.cat((original[:,:start],pruned.unsqueeze(0),original[:,start+1024:]),dim=1),'attention_mask':torch.ones((1,seq-768),dtype=torch.long)}
  adapter.state.update(visual_calls=1,selector_calls=1,greedy_calls=0,splits=[256],visual=visual,pruned=pruned,inputs=inputs,start=start,deepstack=[visual+float(i) for i in range(3)])
  return adapter,kw,ids
 def test_real_selected_rows_drive_p3_and_three_streams(self):
  a,kw,ids=self.make();_,out=a._prefill(None,(),kw);a._observe_prefill(None,(),out)
  self.assertEqual(a.state['spatial'].tolist(),ids.tolist());self.assertEqual(out['position_ids'].shape,(3,1,262));self.assertEqual(out['visual_pos_masks'].sum().item(),256)
  for i in range(3):self.assertTrue(torch.equal(out['deepstack_visual_embeds'][i],a.state['deepstack'][i][ids]))
  self.assertEqual(a.state['selection_mode'],'author_div');self.assertTrue(a.state['actual_injection_verified'])
 def test_wrong_prefill_or_ambiguous_row_fails(self):
  a,kw,_=self.make();kw['inputs_embeds'][0,3,0]+=1
  with self.assertRaisesRegex(ValueError,'prefill'):a._prefill(None,(),kw)
  a,kw,_=self.make();a.state['visual'][1]=a.state['visual'][0]
  with self.assertRaisesRegex(ValueError,'ambiguous'):a._prefill(None,(),kw)
if __name__=='__main__':unittest.main()
