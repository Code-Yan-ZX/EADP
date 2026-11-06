import unittest
import json,tempfile,hashlib,sys
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import torch
from worker import selected_indices,validate_record,persist_generation,preflight,check_calibration_rows
class Tests(unittest.TestCase):
 def test_decode_failure_preserves_full_raw_ids(self):
  class Broken:
   def decode(self,*args,**kwargs):raise RuntimeError('decode failure')
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'generated.json'
   with self.assertRaisesRegex(RuntimeError,'decode failure'):persist_generation(path,torch.tensor([[11,12,13,14]]),2,Broken())
   self.assertEqual(json.loads(path.read_text()),{'prompt_length':2,'full_sequence_ids':[11,12,13,14],'generated_token_ids':[13,14]})
 def test_invalid_prompt_length_also_preserves_raw_ids(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'generated.json'
   with self.assertRaisesRegex(ValueError,'invalid prompt length'):persist_generation(path,torch.tensor([[11,12]]),3,None)
   self.assertEqual(json.loads(path.read_text())['full_sequence_ids'],[11,12])
 def test_changed_prompt_also_preserves_raw_ids(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'generated.json'
   with self.assertRaisesRegex(ValueError,'prompt changed'):persist_generation(path,torch.tensor([[11,12,13]]),2,None,[11,99])
   self.assertEqual(json.loads(path.read_text())['full_sequence_ids'],[11,12,13])
 def test_384_identity_inventory_and_mismatch_fail_closed(self):
  rows=[];refs={};anchors=[]
  for i in range(384):
   message=[{'type':'text','value':str(i)}];h=hashlib.sha256(json.dumps(message,ensure_ascii=False,sort_keys=True).encode()).hexdigest();row={'task':('A','B','C','A')[i%4],'qid':str(i),'group_id':str(i),'split':'DEV','message':message,'message_sha256':h,'image_sha256':'x'};ref={'processor':{'input_ids':i},'image_start':4};rows.append(row);refs[(row['task'],str(i))]=ref;
   if i<4:anchors.append({k:row[k] for k in ('task','qid','group_id','message_sha256','image_sha256')}|ref)
  module=SimpleNamespace(load_accepted_e1_references=lambda *_:(rows,refs))
  with patch.dict(sys.modules,{'input_preflight':module}),patch('worker.calibration_gate'):
   chosen,_=preflight({'sha256':{},'anchors':anchors});self.assertEqual(len(chosen),4)
   broken=[dict(x) for x in anchors];broken[3]['processor']={'input_ids':-1}
   with self.assertRaisesRegex(ValueError,'frozen input mismatch'):preflight({'sha256':{},'anchors':broken})
   with self.assertRaisesRegex(ValueError,'denominator'):preflight({'sha256':{},'anchors':anchors[:-1]})
 def test_calibration_rows_reject_tampering(self):
  anchor={'cpu_index':0,'cpu_K':2,'cpu_selected_indices':[0,3],'task':'DocVQA_VAL','qid':'7'}
  row={'index':0,'threshold':0.1,'ordered_equal':True,'selected_indices':[0,3],'count':2,'identity':{'task':'DocVQA_VAL','qid':'7'}}
  check_calibration_rows({'anchors':[anchor]},[row])
  for changed in ({**row,'selected_indices':[0,2]},{**row,'threshold':0.3},{**row,'ordered_equal':False},{**row,'count':1}):
   with self.assertRaisesRegex(ValueError,'calibration selection mismatch'):check_calibration_rows({'anchors':[anchor]},[changed])
 def test_launcher_imports_real_worker_functions(self):
  import launch,worker
  self.assertIs(launch.preflight,worker.preflight);self.assertIs(launch.save,worker.write)
 def sample(self):
  n=3;seq=1028;initial=seq-1024+n;d=[[-10]]
  trace=[{'rope_delta':d,'model_rope_delta':d,'layers':[{'key':[1,8,initial+i,128],'value':[1,8,initial+i,128]} for _ in range(36)],'position_ids':None if i==0 else [[[7+i]]]*3,'cache_position':list(range(initial)) if i==0 else [seq+i-1]} for i in range(2)]
  return {'selected_count':n,'selected_indices':[0,1,2],'sequence_length':seq,'visual_rows':n,'ds_rows':[n]*3,'prefill_ds_rows':[n]*3,'prefill_visual_count':n,'prefill_main_match':True,'prefill_text_match':True,'prefill_positions_match':True,'prefill_ds_match':[True]*3,'status':'completed','generated_count':2,'ends_eos':True,'cpu_gpu_selection_equal':True,'hook_cleanup_valid':True,'original_rope_delta':d,'original_max_position':7,'trace':trace}
 def test_success_and_eos(self):
  x=self.sample();self.assertTrue(all(validate_record(x).values()));x['ends_eos']=False;self.assertFalse(validate_record(x)['natural_complete'])
 def test_actual_group_validation(self):
  self.assertEqual(selected_indices([torch.tensor([0,1,2,3,8,9,10,11])]),[0,2])
  for x in ([torch.tensor([0,1,2])],[torch.tensor([0,1,2,7])],[torch.tensor([8,9,10,11,0,1,2,3])]):
   with self.assertRaises(ValueError):selected_indices(x)
 def test_reject_ds_cache_and_trace_tampering(self):
  x=self.sample();x['prefill_ds_match'][2]=False;x['trace'][1]['layers'][3]['key'][-2]=1;x['hook_cleanup_valid']=False;c=validate_record(x);self.assertFalse(c['same_main_text_positions']);self.assertFalse(c['positions_and_kv']);self.assertFalse(c['hooks_removed'])
 def test_reject_cpu_gpu_disagreement(self):
  x=self.sample();x['cpu_gpu_selection_equal']=False;self.assertFalse(validate_record(x)['cpu_gpu_selection_identity'])
 def test_reject_incomplete_calls(self):
  x=self.sample();x['trace'].pop();self.assertFalse(validate_record(x)['calls_match_tokens'])
if __name__=='__main__':unittest.main(verbosity=2)
