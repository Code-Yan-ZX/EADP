import copy,types,unittest
import torch
from capture_guard import capture_one
class Vision(torch.nn.Module):
 def forward(self,x):return (x,[x,x,x])
class Language(torch.nn.Module):
 def forward(self,**kwargs):return kwargs['inputs_embeds']
class Model(torch.nn.Module):
 def __init__(self):
  super().__init__();self.visual=Vision();self.model=torch.nn.Module();self.model.language_model=Language();self.model.rope_deltas=torch.tensor([[0]])
 def generate(self,**kwargs):
  self.visual(torch.ones(2));self.model.language_model(inputs_embeds=torch.ones(2),attention_mask=torch.ones(2),position_ids=torch.ones(2),visual_pos_masks=torch.ones(2,dtype=torch.bool),deepstack_visual_embeds=[torch.ones(2)]*3)
  if kwargs.get('fail'):raise RuntimeError('synthetic failure')
  return torch.tensor([[42]])
class Wrapper:
 def __init__(self):self.model=Model();self.pruner=torch.nn.Linear(2,2)
 def _processor_inputs(self):pass
class Adapter:
 def __init__(self,*args):self.state={'prefill_seen':True,'inputs':{'input_ids':torch.ones(2)},'ordered':torch.arange(2)}
 def __enter__(self):return self
 def __exit__(self,*args):pass
class Capture:
 def __init__(self,*args):self.state={'selector':{'result':1},'injection':{'mask':2,'streams':3},**{'merger'+str(i):{'result':i} for i in range(3)}}
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def prepare_operations(self,_):return {'selector':lambda:1,'ds_merger_three_streams':lambda:[0,1,2],'ds_injection':lambda:(2,3)}
class Tests(unittest.TestCase):
 def test_cleanup_and_full_capture(self):
  w=Wrapper();native=types.SimpleNamespace(_greed_select_impl=lambda:None);events=[];row={'task':'D','qid':'1','group_id':'g','image_start':4,'message':[]}
  ops,refs,actual,clean=capture_one(torch,w,native,row,lambda w,m,dataset:w.model.generate(max_new_tokens=8),Adapter,Capture,lambda t,x:copy.deepcopy(x),events.append)
  self.assertTrue(clean());self.assertEqual(events[0]['output_ids'],[[42]]);self.assertEqual(events[-1],{'status':'capture_cleanup','passed':True,'capture_completed':True});self.assertEqual(set(actual),{'visual','prefill','rope_delta','processor','ordered'});self.assertNotIn('generate',w.model.__dict__)
 def test_exception_also_cleans(self):
  w=Wrapper();native=types.SimpleNamespace(_greed_select_impl=lambda:None);events=[];row={'task':'D','qid':'1','group_id':'g','image_start':4,'message':[]}
  with self.assertRaisesRegex(RuntimeError,'synthetic failure'):capture_one(torch,w,native,row,lambda w,m,dataset:w.model.generate(fail=True),Adapter,Capture,lambda t,x:copy.deepcopy(x),events.append)
  self.assertEqual(events[-1],{'status':'capture_cleanup','passed':True,'capture_completed':False});self.assertNotIn('generate',w.model.__dict__);self.assertFalse(w.model.visual._forward_hooks);self.assertFalse(w.model.model.language_model._forward_pre_hooks)
 def test_trainable_merger_requires_real_inference_mode(self):
  from contracts import run_jobs,COMPONENTS
  from test_contracts import Event,schedule
  merger=torch.nn.Linear(2,2);x=torch.ones(2);ops={k:(lambda:merger(x)) for k in COMPONENTS};refs={k:merger(x).detach() for k in COMPONENTS};s=schedule()
  with self.assertRaisesRegex(ValueError,'inference mode required'):run_jobs(s,s,ops,refs,Event,lambda:None,torch.equal,lambda x:str(x),lambda _:None,lambda:True,torch.is_inference_mode_enabled,lambda:True)
  with torch.inference_mode():result=run_jobs(s,s,ops,refs,Event,lambda:None,torch.equal,lambda x:str(x),lambda _:None,lambda:True,torch.is_inference_mode_enabled,lambda:True)
  self.assertEqual(result['full_outputs_checked'],69)
if __name__=='__main__':unittest.main(verbosity=2)
