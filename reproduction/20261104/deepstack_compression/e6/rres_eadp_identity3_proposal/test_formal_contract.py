import copy,tempfile,unittest
from pathlib import Path
import formal_contract,identity_core
from test_identity_core import fixture
class Tests(unittest.TestCase):
 def test_both_arms_and_tampering(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'image';p.write_bytes(b'image')
   for arm in ('rres','eadp'):
    a=fixture(arm,p);ref=a['ref'];ref.update(task='x',qid='1',group_id='g',image_sha256=a['row']['image_sha256'],image_start=2,eos_ids=[151645,151643]);r=identity_core.finish(**a);r['checks']['hooks_and_methods_restored']=True;r['generation_observation']={'explicit_kwargs':formal_contract.RECIPE.copy(),'effective_generation_config':{**formal_contract.RECIPE,'eos_token_id':[151645,151643]},'prepare_calls':1};r['first_criteria_length']=len(a['prompt'])+1 if arm=='rres' else 1;r['criteria_calls']=1
    raw={'full_sequence':[(a['prompt'] if arm=='rres' else [])+a['output']]}
    self.assertTrue(formal_contract.check(r,raw,ref,a['decode']))
    for mutation in ('ds','positions','kwargs','first','raw'):
     x=copy.deepcopy(r);z=copy.deepcopy(raw)
     if mutation=='ds':x['injected_deepstack'][0]['sha256']='bad'
     if mutation=='positions':x['prefill_position_ids'][0][0][0]=100
     if mutation=='kwargs':x['generation_observation']['effective_generation_config']['top_k']=4
     if mutation=='first':x['first_criteria_length']+=1
     if mutation=='raw':z['full_sequence'][0][-1]=8
     with self.assertRaises(ValueError):formal_contract.check(x,z,ref,a['decode'])
