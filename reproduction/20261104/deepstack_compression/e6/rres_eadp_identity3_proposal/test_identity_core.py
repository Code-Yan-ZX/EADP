import hashlib,sys,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT/'e6/rres_eadp_timing_pilot_v4'))
import identity_core as c
from process_core import processor_evidence
class Decode:
 @staticmethod
 def validate_trace(trace,output,compact,delta):
  if trace!=[{'compact':compact,'delta':delta}]:raise ValueError('wrong KV')
def fixture(arm,path):
 n=260 if arm=='rres' else 1028;prompt=list(range(n));inputs={'input_ids':torch.tensor([prompt]),'attention_mask':torch.ones((1,n),dtype=torch.int64),'pixel_values':torch.arange(16,dtype=torch.float32).reshape(4,4),'image_grid_thw':torch.tensor([[1,32,32]])}
 count=256 if arm=='rres' else 1024;visual=torch.ones((count,3));deep=[visual*(i+1) for i in range(3)];injected=[x[:256] for x in deep]
 pos=torch.zeros((3,1,n if arm=='rres' else n-768),dtype=torch.int64);mask=torch.zeros((1,pos.shape[-1]),dtype=torch.bool);mask[:,2:258]=True;delta=torch.tensor([[2]])
 embed=torch.ones((1,pos.shape[-1],3));attention=torch.ones((1,pos.shape[-1]),dtype=torch.int64)
 logits=torch.ones((1,8));state={'visual':visual,'deepstack':deep,'injected_ds':injected,'prefill_position_ids':pos,'delta':delta,'prefill_visual_pos_masks':mask,'prefill_inputs_embeds':embed,'prefill_attention_mask':attention,'block_calls':[1,1],'merger_calls':1,'ds_calls':[1,1,1],'logits':logits}
 row={'task':'x','qid':'1','group_id':'g','image_path':str(path),'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'image_start':2};out=[9];ref={'processor':processor_evidence(torch,inputs),'output_ids':out,'answer':'a','main_visual':c.meta(torch,visual)}
 if arm=='rres':ref.update(prefill_position_ids=c.meta(torch,pos),delta=[[2]],deepstack=[c.meta(torch,x) for x in deep],injected_ds=[c.meta(torch,x) for x in injected],prefill_inputs_embeds=c.meta(torch,embed),prefill_attention_mask=c.meta(torch,attention),prefill_logits=c.meta(torch,logits),block_calls=[1,1],merger_calls=1,ds_calls=[1,1,1]);sel=None
 else:ref.update(prefill_position_ids=pos.tolist(),adjusted_delta=[[2]],full_deepstack=[c.meta(torch,x) for x in deep],injected_deepstack=[c.meta(torch,x) for x in injected],injected_mask=mask.tolist(),ordered_ids=list(range(256)),prefill_embeds=c.meta(torch,embed),prefill_mask=c.meta(torch,attention),block_calls=[1,1],main_merger_calls=1,ds_calls=[1,1,1]);sel=list(range(256))
 trace=[{'compact':n if arm=='rres' else n-768,'delta':2}]
 return dict(torch=torch,arm=arm,row=row,ref=ref,inputs=inputs,prompt=prompt,output=out,response='a',state=state,trace=trace,selected=sel,decode=Decode)
class Tests(unittest.TestCase):
 def test_identity_and_independent_mismatches(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'image';path.write_bytes(b'image')
   for arm in ('rres','eadp'):
    a=fixture(arm,path);self.assertEqual(c.finish(**a)['status'],'passed')
    a['trace'][0]['compact']+=1;r=c.finish(**a);self.assertEqual(r['status'],'failed');self.assertFalse(r['checks']['all_36_layer_kv_positions'])
    a=fixture(arm,path);a['state']['injected_ds'][0][0,0]+=1;self.assertFalse(c.finish(**a)['checks']['three_stream_injection'])
    for field,check in [('prefill_inputs_embeds','prefill_embeds'),('prefill_attention_mask','prefill_attention')]:
     a=fixture(arm,path);a['state'][field][0,0]+=1;self.assertFalse(c.finish(**a)['checks'][check])
    a=fixture(arm,path);a['state']['ds_calls'][0]=2;self.assertFalse(c.finish(**a)['checks']['visual_execution_counts'])
    if arm=='rres':
     a=fixture(arm,path);a['state']['logits'][0,0]+=1;self.assertFalse(c.finish(**a)['checks']['prefill_logits'])
     a=fixture(arm,path);del a['state']['logits']
     with self.assertRaises(KeyError):c.finish(**a)
    if arm=='eadp':
     a=fixture(arm,path);a['selected'][0]=255;self.assertFalse(c.finish(**a)['checks']['ordered_selection'])
 def test_eadp_diagnostic_hooks_follow_the_actual_adapter(self):
  events=[]
  class Adapter:
   def __init__(self,*args):self.state={'ordered':torch.arange(256)}
   def __enter__(self):events.append('adapter_enter');return self
   def __exit__(self,*args):events.append('adapter_exit')
  class Capture:
   def __init__(self,*args):self.state={}
   def __enter__(self):events.append('capture_enter');return self
   def __exit__(self,*args):events.append('capture_exit')
  class Trace:
   def __init__(self,*args):self.trace=[]
   def __enter__(self):events.append('trace_enter');return self
   def __exit__(self,*args):events.append('trace_exit')
  class CaptureModule:pass
  class DecodeModule:pass
  CaptureModule.Capture=Capture;DecodeModule.DecodeObserver=Trace
  class Model:pass
  class Wrapper:pass
  wrapper=Wrapper();wrapper.model=Model();wrapper.model.model=Model()
  class Qwen:
   class runner:
    @staticmethod
    def normalize(arm,raw,prompt):return raw
   @staticmethod
   def request(**kw):
    with kw['adapter_type'](1,2):
     events.append('request');kw['persist'](torch.tensor([[7]]))
    return {'processor_inputs':{'input_ids':torch.tensor([[1,2]])},'response':'a','generation':{},'times':{'first_criteria_length':1,'criteria_calls':1}}
  row={'task':'x','qid':'1','group_id':'g'};capture=[]
  with patch('identity_core.finish',return_value={'checks':{},'errors':[]}):
   result=c.run_one(torch=torch,wrapper=wrapper,module=None,pruner=None,row=row,ref={},arm='eadp',qwen_request=Qwen,adapter_type=Adapter,criteria_type=None,fixed_resolution=None,author_generate=None,capture_module=CaptureModule,decode=DecodeModule,persist=capture.append)
  self.assertEqual(events,['adapter_enter','capture_enter','trace_enter','request','trace_exit','capture_exit','adapter_exit']);self.assertEqual(result['status'],'passed');self.assertEqual(len(capture),1)
