import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import process_core
class Tensor:
 def __init__(self,x):self.x=x
 def detach(self):return self
 def cpu(self):return self
 def tolist(self):return self.x
class Core:rope_deltas=22
class Model:
 def __init__(self):self.model=Core()
class Wrapper:
 def __init__(self):self.model=Model()
class Tests(unittest.TestCase):
 def test_matches_and_mismatches_are_retained(self):
  item={'phase':'measured','replicate':0,'task':'x','qid':'1','group_id':'g'};job={'arm':'rres','task':'x','requests':[item,item]};row={**item};ref={'output_ids':[7],'answer':'a'};rows={('x','1'):row};refs={('rres','x','1'):ref};n=[0]
  def request(**kw):
   self.assertIsNone(kw['wrapper'].model.model.rope_deltas);n[0]+=1;kw['persist'](Tensor([[1,2,7 if n[0]==1 else 8]]));return {'response':'a','times':{},'generation':{},'rres_prompt_tensor':Tensor([[1,2]])}
  with tempfile.TemporaryDirectory() as d,patch('process_core.qwen_request.request',request):
   result=process_core.run_process(job,rows,refs,torch=None,wrapper=Wrapper(),module=None,pruner=None,adapter_type=None,criteria_type=None,fixed_resolution=None,author_generate=None,out=d)
   events=[json.loads(x) for x in (Path(d)/'journal.jsonl').read_text().splitlines()]
   self.assertEqual(result['identity_mismatches'],1);self.assertEqual(len(events),4);self.assertFalse(events[-1]['expected_tokens_match']);self.assertTrue((Path(d)/'raw_0001.json').is_file())
 def test_failure_after_raw_is_retained(self):
  item={'phase':'warmup','replicate':0,'task':'x','qid':'1','group_id':'g'};job={'arm':'eadp','task':'x','requests':[item]}
  def request(**kw):kw['persist'](Tensor([[7]]));raise RuntimeError('decode failed')
  with tempfile.TemporaryDirectory() as d,patch('process_core.qwen_request.request',request):
   with self.assertRaises(RuntimeError):process_core.run_process(job,{('x','1'):item},{('eadp','x','1'):{}},torch=None,wrapper=Wrapper(),module=None,pruner=None,adapter_type=None,criteria_type=None,fixed_resolution=None,author_generate=None,out=d)
   event=json.loads((Path(d)/'journal.jsonl').read_text().splitlines()[-1]);self.assertEqual(event['event'],'failed');self.assertTrue(event['raw_persisted'])
