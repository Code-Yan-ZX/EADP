import copy,json,unittest
from pathlib import Path
import torch
from worker import profile
from test_runner import Event

class WorkerTests(unittest.TestCase):
 def setUp(self):
  self.frozen=json.loads((Path(__file__).parent/'schedule001.json').read_text());self.schedule=self.frozen['schedule'];self.identities=self.frozen['input_identities']
  self.rows=[dict(x,split='DEV') for x in self.identities];self.count={'capture':0,'preflight':0,'journal':0}
 def preflight(self,row,ref):self.count['preflight']+=1
 def capture(self,row):
  self.count['capture']+=1
  return {k:(lambda:None) for k in self.schedule['components']}
 def journal(self,value):self.count['journal']+=1
 def run_profile(self,rows,identities):return profile(torch,self.schedule,self.schedule,rows,identities,self.preflight,self.capture,Event,lambda:None,self.journal)
 def test_capture_once_per_group_and_all_component_measurements(self):
  result=self.run_profile(self.rows,self.identities)
  self.assertEqual(result['measurements'],1920);self.assertEqual(self.count,{'capture':32,'preflight':32,'journal':1920})
 def test_modified_question_identity_rejected_before_capture(self):
  rows=copy.deepcopy(self.rows);rows[-1]['message_sha256']='tampered'
  with self.assertRaisesRegex(ValueError,'identity mismatch'):self.run_profile(rows,self.identities)
  self.assertEqual(self.count['capture'],0)
if __name__=='__main__':unittest.main()

class CaptureOrchestrationTests(unittest.TestCase):
 def test_replay_validation_runs_after_all_hooks_removed(self):
  from types import SimpleNamespace,ModuleType
  from unittest.mock import patch
  from worker import capture_one
  flags={'adapter':False,'capture':False};events=[]
  class Adapter:
   def __init__(self,*args):self.state={'prefill_seen':True}
   def __enter__(self):flags['adapter']=True;return self
   def __exit__(self,*args):flags['adapter']=False
  class Capturer:
   def __init__(self,*args):pass
   def __enter__(self):flags['capture']=True;return self
   def __exit__(self,*args):flags['capture']=False
   def prepare_operations(self,start):
    if flags['adapter'] or flags['capture']:raise AssertionError('hooks still present')
    return {'selector':lambda:None,'ds_merger_three_streams':lambda:None,'ds_injection':lambda:None}
  module=ModuleType('light_adapter');module.LightAdapter=Adapter
  def generate(**kwargs):
   self.assertEqual(kwargs['max_new_tokens'],1);return torch.tensor([[42]])
  model=SimpleNamespace(generate=generate);wrapper=SimpleNamespace(model=model)
  row={'task':'DEV','qid':'1','group_id':'g','image_start':4,'message':[]}
  with patch.dict('sys.modules',{'light_adapter':module}),patch('worker.CaptureInputs',Capturer):
   ops=capture_one(torch,wrapper,None,row,lambda w,m,dataset:w.model.generate(max_new_tokens=2048),events.append)
  self.assertIs(wrapper.model.generate,generate);self.assertEqual(events[0]['output_ids'],[[42]]);self.assertEqual(len(ops),3)

class PilotTests(unittest.TestCase):
 def test_pilot_is_exact_frozen_first_image(self):
  from runner import pilot_schedule
  frozen=json.loads((Path(__file__).parent/'schedule001.json').read_text())['schedule'];pilot=pilot_schedule(frozen)
  self.assertEqual(pilot['jobs'],frozen['jobs'][:60]);self.assertEqual(pilot['images'],frozen['images'][:1]);self.assertEqual(pilot['stage'],'pilot')
  damaged=copy.deepcopy(frozen);damaged['jobs'][0]['group_id']='wrong'
  with self.assertRaises(ValueError):pilot_schedule(damaged)
