"""Black-box CPU tests of the actual v2 process path; mock only model request."""
import hashlib,importlib.util,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent;B=HERE.parents[1];C=B/'e6'/os.environ.get('PAIRED_TIMING_CANDIDATE','rres_eadp_timing_pilot_v2')
@unittest.skipUnless(C.is_dir(),'v2 candidate unavailable')
class TestProcess(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import torch
  cls.torch=torch;sys.path.insert(0,str(C))
  s=importlib.util.spec_from_file_location('_independent_v2_process',C/'process_core.py');cls.mod=importlib.util.module_from_spec(s);s.loader.exec_module(cls.mod)
 def setup_case(self,td,arm):
  t=self.torch;image=Path(td)/'image.bin';image.write_bytes(b'image');image_hash=hashlib.sha256(b'image').hexdigest()
  inputs={'input_ids':t.tensor([[1,2]],dtype=t.int64),'attention_mask':t.tensor([[1,1]],dtype=t.int64),'pixel_values':t.tensor([[1.,2.]],dtype=t.float32),'image_grid_thw':t.tensor([[1,2,2]],dtype=t.int64)}
  meta={k:{'dtype':str(v.dtype),'shape':list(v.shape),'sha256':hashlib.sha256(v.numpy().tobytes()).hexdigest()} for k,v in inputs.items()}
  item={'task':'t','qid':'1','group_id':'g','phase':'measured','replicate':0};row={**item,'image_path':str(image),'image_sha256':image_hash};ref={'processor':meta,'output_ids':[9,10],'answer':'text'}
  class Core:rope_deltas=4
  class Model:model=Core()
  class Wrapper:model=Model()
  return inputs,item,row,ref,Wrapper()
 def invoke(self,td,arm='rres',mutation=None):
  t=self.torch;inputs,item,row,ref,w=self.setup_case(td,arm)
  if mutation=='missing_row':rows={}
  else:rows={('t','1'):row}
  def request(**kw):
   raw=t.tensor([[1,2,9,10]]) if arm=='rres' else t.tensor([[9,10]])
   kw['persist'](raw)
   if mutation=='wrong_processor':inputs['pixel_values'][0,0]=7
   if mutation=='wrong_image':Path(row['image_path']).write_bytes(b'changed')
   first=(3 if arm=='rres' else 1)+(1 if mutation=='wrong_first' else 0)
   return {'processor_inputs':inputs,'response':'text','times':{'first_criteria_length':first,'criteria_calls':1 if mutation=='wrong_count' else 2},'generation':{}}
  job={'task':'t','arm':arm,'requests':[item]};out=Path(td)/'out';out.mkdir()
  with patch.object(self.mod.qwen_request,'request',request):
   if mutation=='missing_row':
    with self.assertRaises(KeyError):self.mod.run_process(job,rows,{(arm,'t','1'):ref},torch=t,wrapper=w,module=None,pruner=None,adapter_type=None,criteria_type=None,fixed_resolution=None,author_generate=None,out=out)
    return [json.loads(x) for x in (out/'journal.jsonl').read_text().splitlines()],None
   result=self.mod.run_process(job,rows,{(arm,'t','1'):ref},torch=t,wrapper=w,module=None,pruner=None,adapter_type=None,criteria_type=None,fixed_resolution=None,author_generate=None,out=out)
   return [json.loads(x) for x in (out/'journal.jsonl').read_text().splitlines()],result
 def test_both_arms_correct(self):
  for arm in ('rres','eadp'):
   with self.subTest(arm=arm),tempfile.TemporaryDirectory() as td:
    events,result=self.invoke(td,arm);self.assertEqual(result['identity_mismatches'],0);self.assertTrue(all(events[-1]['checks'].values()))
 def test_all_processor_and_endpoint_mismatches_retained(self):
  for mutation,field in [('wrong_processor','processor_match'),('wrong_image','image_match'),('wrong_first','first_criteria_length_match'),('wrong_count','criteria_count_match')]:
   with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as td:
    events,result=self.invoke(td,mutation=mutation);self.assertEqual(result['identity_mismatches'],1);self.assertFalse(events[-1]['checks'][field]);self.assertTrue((Path(td)/'out/raw_0000.json').is_file())
 def test_missing_row_leaves_failure_event(self):
  with tempfile.TemporaryDirectory() as td:
   events,_=self.invoke(td,mutation='missing_row');self.assertEqual([e['event'] for e in events],['started','failed']);self.assertFalse(events[-1]['raw_persisted'])
if __name__=='__main__':unittest.main()
