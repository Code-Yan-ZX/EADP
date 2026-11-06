"""CPU-only actual run_arm success/restore on a synthetic fake model."""
import sys,unittest,tempfile,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
B=Path(__file__).resolve().parents[2];P=B/'e6/rres_k256_pilot_v2';sys.path.insert(0,str(P));import arm
class Model:
 def __init__(self,pos):self.model=SimpleNamespace(get_rope_index=lambda *_a,**_k:(pos,torch.tensor([[-240]])));self.generation_config=SimpleNamespace(eos_token_id=[151645,151643])
 def modules(self):return []
 def generate(self,**kw):return torch.cat((kw['input_ids'],torch.tensor([[151645]])),dim=1)
class Wrapper:
 def __init__(self,inputs,pos):self.model=Model(pos);self.inputs=inputs
 def _processor_inputs(self,_messages):return self.inputs
 def generate_inner(self,message,dataset=None):
  x=self._processor_inputs(message);self.model.generate(**x,do_sample=False,max_new_tokens=2048);return ''
class FakeCapture:
 def __init__(self,_torch,_wrapper):
  v=torch.ones((256,2));ds=[v*(i+2) for i in range(3)];mask=torch.zeros((1,260),dtype=torch.bool);mask[:,2:258]=True;emb=torch.zeros((1,260,2));emb[mask]=v;pos=torch.arange(260).repeat(3,1).unsqueeze(1)
  self.state={'block_calls':[1,1],'merger_calls':1,'ds_calls':[1,1,1],'visual':v,'deepstack':ds,'prefill_position_ids':pos,'delta':torch.tensor([[-240]]),'prefill_attention_mask':torch.ones((1,260),dtype=torch.long),'prefill_visual_pos_masks':mask,'prefill_inputs_embeds':emb,'injected_ds':[x.clone() for x in ds],'logits':torch.ones((1,5))}
 def __enter__(self):return self
 def __exit__(self,*a):pass
class FakeDecode:
 class DecodeObserver:
  def __init__(self,*a):self.trace=[{'dummy':True}]
  def __enter__(self):return self
  def __exit__(self,*a):pass
 @staticmethod
 def validate_trace(trace,ids,n,delta):assert len(trace)==1 and ids==[151645] and n==260 and delta==-240
class Tests(unittest.TestCase):
 def test_actual_arm_end_to_end_success(self):
  inputs={'input_ids':torch.arange(260).reshape(1,-1),'attention_mask':torch.ones((1,260),dtype=torch.long),'pixel_values':torch.ones((1,2)),'image_grid_thw':torch.tensor([[1,32,32]])};pos=torch.arange(260).repeat(3,1).unsqueeze(1);w=Wrapper(inputs,pos);module=SimpleNamespace(QWEN3_FIXED_RESOLUTION=1024);ref={'low_processor':{k:arm.meta(v) for k,v in inputs.items()},'position_ids':arm.meta(pos),'rope_delta':[[-240]],'image_start':2};row={'task':'t','qid':'1','group_id':'g','image_sha256':'i','message_sha256':'m','message':'x'}
  with tempfile.TemporaryDirectory() as tmp:
   raw=Path(tmp)/'raw.json'
   with patch.object(arm,'Capture',FakeCapture):r=arm.run_arm(torch,w,module,row,ref,raw,FakeDecode)
   self.assertEqual(r['output_ids'],[151645]);self.assertTrue(r['hooks_and_methods_restored']);self.assertEqual(json.loads(raw.read_text())['full_sequence'][0][-1],151645)
  self.assertEqual(module.QWEN3_FIXED_RESOLUTION,1024)
if __name__=='__main__':unittest.main()
