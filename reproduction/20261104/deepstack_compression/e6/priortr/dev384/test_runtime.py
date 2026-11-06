import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from runtime import invoker

class Inputs(dict):
 def to(self,*_):return self
class Model:
 def __init__(self):self.swaps=0;self.model=SimpleNamespace(rope_deltas=torch.tensor([[1]]));self._rope_position_offset=0;self.generation_config=SimpleNamespace(eos_token_id=[2])
 def _replace_text_model(self):self.swaps+=1
 def _create_strategy(self,name):return name
 def generate(self,**_):return torch.tensor([[8,9,2]])
class Trace:
 def __init__(self,model,forced=False):self.calls=[{'step':0}];self.layers=[{'layer':0}];self.injections=[1,2,3];self.prunes=[];self.arrays={}
 def __enter__(self):return self
 def __exit__(self,*_):pass
 def save(self,k,t):self.arrays[k]=t.detach().clone()
class Config:
 def __init__(self,**kw):self.kw=kw
class Tok:
 def decode(self,ids,**kw):return 'synthetic'
class RuntimeTests(unittest.TestCase):
 def test_swap_once_event_and_archive_policies(self):
  model=Model();proc=SimpleNamespace(tokenizer=Tok());events=[]
  def prepare(*_):return Inputs(input_ids=torch.tensor([[8,9]]),attention_mask=torch.ones((1,2),dtype=torch.long),pixel_values=torch.ones((1,2)),image_grid_thw=torch.tensor([[1,1,1]]))
  row={'task':'DocVQA_VAL','qid':'q','group_id':'g','image_sha256':'i','message_sha256':'m','message':[]}
  with tempfile.TemporaryDirectory() as d:
   call=invoker(torch=torch,model=model,processor=proc,VTRConfig=Config,Trace=Trace,prepare=prepare,arrays_dir=Path(d))
   a=call(row,'stock_native',events.append,True);b=call(row,'priortr_k256',events.append,False);call(row,'author_default',events.append,False)
   self.assertEqual(model.swaps,1);self.assertEqual(a['output_ids'],[2]);self.assertTrue(a['eos']);self.assertEqual(a['arrays']['kind'],'full_arrays');self.assertTrue(Path(a['arrays']['path']).exists());self.assertEqual(b['arrays']['kind'],'summary_only');self.assertEqual(len(events),3)
   with self.assertRaisesRegex(ValueError,'already patched'):call(row,'stock_native',events.append,False)
if __name__=='__main__':unittest.main()
