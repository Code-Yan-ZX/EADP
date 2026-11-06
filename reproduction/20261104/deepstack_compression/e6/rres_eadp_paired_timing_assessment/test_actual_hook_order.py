"""CPU synthetic forwards through the actual adapter and identity Capture hooks."""
import importlib.util,types,unittest
from pathlib import Path
import torch
B=Path(__file__).resolve().parents[2]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
adapter=load('_independent_real_budget_adapter',B/'e4_c_budget_pilot_v2/adapter.py')
capture=load('_independent_real_rres_capture',B/'e6/rres_k256_pilot_v2/capture.py')
class Vision(torch.nn.Module):
 def __init__(self):
  super().__init__();self.blocks=torch.nn.ModuleList([torch.nn.Identity()]);self.merger=torch.nn.Identity();self.deepstack_merger_list=torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
 def forward(self,x):
  for b in self.blocks:x=b(x)
  return self.merger(x),[d(x+i) for i,d in enumerate(self.deepstack_merger_list)]
class Language(torch.nn.Module):
 def forward(self,**kw):return kw
class Inner:
 def __init__(self):self.language_model=Language();self.rope_deltas=None
 def get_rope_index(self,ids,grid,_unused,attention_mask=None):
  return torch.arange(ids.shape[1]).repeat(3,1,1),torch.tensor([[-1]])
class Model:
 def __init__(self):self.visual=Vision();self.model=Inner();self.lm_head=torch.nn.Identity()
class Wrapper:
 def __init__(self):self.model=Model()
 def _processor_inputs(self):return {'input_ids':torch.arange(1028).reshape(1,-1),'attention_mask':torch.ones((1,1028),dtype=torch.long),'image_grid_thw':torch.tensor([[1,64,64]])}
class Tests(unittest.TestCase):
 def exercise(self,exception=False):
  w=Wrapper();original=w._processor_inputs;calls=[]
  def greedy(*a,**kw):return (torch.arange(255,-1,-1),)
  pruner=types.SimpleNamespace(_greed_select_impl=greedy)
  with self.assertRaisesRegex(RuntimeError,'deliberate') if exception else __import__('contextlib').nullcontext():
   with adapter.BudgetLightAdapter(torch,w,pruner,2,256) as a:
    with capture.Capture(torch,w) as c:
     inputs=w._processor_inputs();v,deep=w.model.visual(torch.ones((1024,3)));pruner._greed_select_impl()
     result=w.model.model.language_model(inputs_embeds=torch.zeros((1,260,3)),attention_mask=torch.ones((1,260),dtype=torch.long),position_ids=torch.full((3,1,260),-1),visual_pos_masks=None,deepstack_visual_embeds=None)
     s=c.state;self.assertTrue(a.state['prefill_seen']);self.assertEqual(a.state['greedy_calls'],1)
     expected=torch.cat((torch.arange(2),2+torch.arange(256),torch.arange(1026,1028)))
     self.assertTrue(torch.equal(s['prefill_position_ids'],torch.arange(1028).repeat(3,1,1).index_select(-1,expected)))
     self.assertTrue(torch.equal(s['prefill_visual_pos_masks'],result['visual_pos_masks']));self.assertEqual(int(s['prefill_visual_pos_masks'].sum()),256)
     for observed,original_stream in zip(s['injected_ds'],deep):self.assertTrue(torch.equal(observed,original_stream[:256]))
     if exception:raise RuntimeError('deliberate')
  self.assertEqual(w._processor_inputs,original);self.assertIs(pruner._greed_select_impl,greedy)
  self.assertFalse(w.model.model.language_model._forward_pre_hooks);self.assertFalse(w.model.visual._forward_hooks)
 def test_capture_observes_adapter_rewritten_prefill(self):self.exercise()
 def test_hook_restoration_on_error(self):self.exercise(True)
if __name__=='__main__':unittest.main()
