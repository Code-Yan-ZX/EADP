"""Execute draft hooks around a synthetic CPU author-shaped pipeline; no model."""
import os,unittest
from types import SimpleNamespace
import torch
from worker_draft import run_arm,baseline_capture
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
ids=torch.tensor([[8,9]+[99]*1024+[10,11]],dtype=torch.long)
visual=torch.arange(1024*8,dtype=torch.float32).reshape(1024,8)
original_ids=torch.arange(255,-1,-1,dtype=torch.long).view(1,256)
class Vision(torch.nn.Module):
 def forward(self):return visual,[visual+1,visual+2,visual+3]
class Language(torch.nn.Module):
 def forward(self,**kwargs):
  output=kwargs['inputs_embeds'].clone()
  if kwargs['visual_pos_masks'] is not None:
   for layer in kwargs['deepstack_visual_embeds']:output[kwargs['visual_pos_masks']]+=layer
  return output
class Pruner(torch.nn.Module):
 def __init__(self,native):super().__init__();self.native=native
 def forward(self,x):
  selected,quota=self.native._greed_select_impl(None,None,256)
  return x[selected[0].sort().values],[256]
class SyntheticModel(torch.nn.Module):
 def __init__(self):
  super().__init__();self.visual=Vision();self.model=SimpleNamespace(language_model=Language(),rope_deltas=torch.zeros((1,1),dtype=torch.long));self.config=SimpleNamespace(image_token_id=99)
 def get_input_embeddings(self):return lambda t:torch.arange(t.numel()*8,dtype=torch.float32).reshape(1,t.numel(),8)
 def generate(self,inputs_embeds,attention_mask,**kwargs):
  n=inputs_embeds.shape[1]
  self.model.language_model(inputs_embeds=inputs_embeds,attention_mask=attention_mask,position_ids=torch.zeros(3,1,n,dtype=torch.long),cache_position=torch.arange(n),visual_pos_masks=None,deepstack_visual_embeds=None)
  return torch.tensor([[7,8]])
class Wrapper:
 def __init__(self,native):self.model=SyntheticModel();self.pruner=Pruner(native)
 def _processor_inputs(self,*a,**k):return dict(input_ids=ids,attention_mask=torch.ones_like(ids),pixel_values=torch.ones(1,8),image_grid_thw=torch.tensor([[1,64,64]]))
def method(w,message,dataset):
 inputs=w._processor_inputs(message);v,_=w.model.visual();selected,sizes=w.pruner(v)
 original=w.model.get_input_embeddings()(inputs['input_ids']);embedding=torch.cat([original[:,:2],selected.unsqueeze(0),original[:,1026:]],dim=1)
 attention=torch.ones((1,260),dtype=torch.long)
 w.model.generate(inputs_embeds=embedding,attention_mask=attention,do_sample=False,max_new_tokens=2048,top_p=.001,top_k=1,temperature=.01,repetition_penalty=1.)
 return 'native answer'
class Tests(unittest.TestCase):
 def setUp(self):
  self.current=original_ids.clone();self.native=SimpleNamespace(_greed_select_impl=lambda *a:(self.current,torch.tensor([256])))
  self.w=Wrapper(self.native);self.saved=[];self.item=dict(index='example',message=[])
 def capture(self,item,arm,response,failure):self.saved.append((arm,response,failure))
 def test_paired_hooks_and_native_response(self):
  off=run_arm(torch,self.w,self.native,method,self.item,'ds_off',None,self.capture);base=baseline_capture(self.w,off)
  on=run_arm(torch,self.w,self.native,method,self.item,'ds_on',base,self.capture)
  self.assertTrue(on['intervention_applied'],on['intervention_error']);self.assertEqual(len(self.saved),2)
  self.assertEqual(self.saved,[('ds_off','native answer',None),('ds_on','native answer',None)])
  self.assertTrue(torch.equal(on['final_prefill']['visual_pos_masks'].sum(),torch.tensor(256)))
  self.assertTrue(torch.equal(on['final_prefill']['deepstack_visual_embeds'][0],(visual+1)[:256]))
  self.assertIsNone(off['final_prefill']['visual_pos_masks'])
 def test_failed_hook_installation_cleans_up_and_preserves_failure(self):
  self.w.pruner.register_forward_hook=lambda *args,**kwargs:(_ for _ in ()).throw(RuntimeError('hook install failure'))
  result=run_arm(torch,self.w,self.native,method,self.item,'ds_off',None,self.capture)
  self.assertIn('hook install failure',result['native_failure']);self.assertEqual(self.saved[-1][0],'ds_off')
  self.assertFalse(self.w.model.visual._forward_hooks);self.assertFalse(self.w.model.model.language_model._forward_pre_hooks)
 def test_selection_difference_saves_native_response_before_failure(self):
  off=run_arm(torch,self.w,self.native,method,self.item,'ds_off',None,self.capture);base=baseline_capture(self.w,off)
  self.current=torch.arange(256,0,-1).view(1,256)
  on=run_arm(torch,self.w,self.native,method,self.item,'ds_on',base,self.capture)
  self.assertEqual(self.saved[-1],('ds_on','native answer',None));self.assertFalse(on['intervention_applied']);self.assertTrue(on['intervention_error'])
  self.assertIsNone(on['final_prefill']['visual_pos_masks'])
if __name__=='__main__':
 r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 assert not torch.cuda.is_initialized();raise SystemExit(0 if r.wasSuccessful() else 1)
