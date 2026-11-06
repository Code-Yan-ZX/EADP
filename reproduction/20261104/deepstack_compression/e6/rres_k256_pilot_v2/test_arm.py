"""Exercise actual arm wrapper ordering with a CPU-only fake model."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import arm
class Context:
 def __init__(self,*_a):self.state={};self.trace=[]
 def __enter__(self):return self
 def __exit__(self,*_a):return False
class Model:
 def __init__(self,inputs,positions):self.model=SimpleNamespace(get_rope_index=lambda *_a,**_k:(positions,torch.tensor([[-240]])));self.inputs=inputs
 def modules(self):return []
 def generate(self,**_kw):return torch.tensor([[1,2,3,4]])
class Wrapper:
 def __init__(self,model,inputs):self.model=model;self.inputs=inputs
 def _processor_inputs(self,_messages):return self.inputs
 def generate_inner(self,message,dataset=None):
  inputs=self._processor_inputs(message);self.model.generate(**inputs,do_sample=False,max_new_tokens=2048);raise RuntimeError('synthetic failure after raw')
class Tests(unittest.TestCase):
 def test_raw_durable_before_postprocessing_failure_and_restored(self):
  inputs={'input_ids':torch.tensor([[1,2]]),'attention_mask':torch.tensor([[1,1]]),'pixel_values':torch.ones((1,2)),'image_grid_thw':torch.tensor([[1,32,32]])};positions=torch.ones((3,1,2),dtype=torch.long);m=Model(inputs,positions);w=Wrapper(m,inputs);module=SimpleNamespace(QWEN3_FIXED_RESOLUTION=1024)
  ref={'low_processor':{k:arm.meta(v) for k,v in inputs.items()},'position_ids':arm.meta(positions),'rope_delta':[[-240]]};row={'task':'task','qid':'1','group_id':'g','message':'x'}
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'raw.json'
   with patch.object(arm,'Capture',Context):
    with self.assertRaisesRegex(RuntimeError,'synthetic failure'):arm.run_arm(torch,w,module,row,ref,p,SimpleNamespace(DecodeObserver=Context))
   self.assertEqual(json.loads(p.read_text())['full_sequence'],[[1,2,3,4]])
  self.assertNotIn('_processor_inputs',w.__dict__);self.assertNotIn('generate',m.__dict__);self.assertEqual(module.QWEN3_FIXED_RESOLUTION,1024)
if __name__=='__main__':unittest.main()
