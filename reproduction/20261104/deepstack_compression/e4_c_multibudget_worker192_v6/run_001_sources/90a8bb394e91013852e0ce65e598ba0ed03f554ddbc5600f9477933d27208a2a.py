import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace,ModuleType
from unittest.mock import patch
import torch
import worker
class PixelModel:
 def __init__(self):self.generation_config=SimpleNamespace(eos_token_id=9)
 def generate(self,**kw):
  result=torch.tensor([[4,5,6,9]])
  kw['stopping_criteria'](torch.tensor([[4,5,6]]),None)
  return result
class Wrapper:
 def __init__(self):self.model=PixelModel()
 def _processor_inputs(self,*args,**kw):return {'input_ids':torch.tensor([[4,5]])}
 def generate_inner_transformers_fixed(self,message,dataset):
  x=self._processor_inputs(message)
  x.update(do_sample=False,max_new_tokens=2048,top_p=.001,top_k=1,temperature=.01,repetition_penalty=1.)
  self.model.generate(**x)
  return 'answer'
class FakeCuda:
 def synchronize(self):pass
 def memory_allocated(self):return 10
 def memory_reserved(self):return 20
 def reset_peak_memory_stats(self):pass
 def max_memory_allocated(self):return 30
 def max_memory_reserved(self):return 40
class Tests(unittest.TestCase):
 def test_pixel_real_call_path_observes_actual_count_and_raw_before_completion(self):
  wrapper=Wrapper();original=wrapper.model.generate;events=[]
  p=SimpleNamespace(compute_merged_keep_indices=lambda:[[1,2,3]])
  orig=wrapper.model.generate
  def gen(**kw):p.compute_merged_keep_indices();return orig(**kw)
  wrapper.model.generate=gen
  value=worker.execute_pixel(wrapper,p,torch,{'message':[],'task':'a'},lambda:None,{'tokens':[6,9],'eos_ids':[9],'selected_count':3},lambda raw:events.append(raw),validate_input=lambda x:None)
  self.assertEqual(value['dynamic_k'],3);self.assertEqual(value['tokens'],[6,9]);self.assertEqual(events,[[[4,5,6,9]]]);self.assertEqual(value['boundaries']['first_criteria_length'],3)
  self.assertIs(wrapper.model.generate,gen)
 def test_observer_disabled_and_enabled_same_output(self):
  w=Wrapper();p=SimpleNamespace(compute_merged_keep_indices=lambda:[[1,2]])
  original=w.model.generate
  def gen(**kw):p.compute_merged_keep_indices();return original(**kw)
  w.model.generate=gen;raw=[];checks=[]
  a=worker.execute_pixel(w,p,torch,{'message':[],'task':'a'},lambda:None,{'tokens':[6,9],'eos_ids':[9],'selected_count':2},lambda x:raw.append(x),False,lambda x:checks.append(x))
  b=worker.execute_pixel(w,p,torch,{'message':[],'task':'a'},lambda:None,{'tokens':[6,9],'eos_ids':[9],'selected_count':2},lambda x:raw.append(x),True,lambda x:checks.append(x))
  self.assertEqual(a['tokens'],b['tokens']);self.assertIsNone(a['dynamic_k']);self.assertEqual(b['dynamic_k'],2);self.assertEqual(len(checks),2)
 def test_actual_processor_and_k_eos_gates_fail_closed(self):
  w=Wrapper();p=SimpleNamespace(compute_merged_keep_indices=lambda:[[1,2]])
  original=w.model.generate
  def gen(**kw):p.compute_merged_keep_indices();return original(**kw)
  w.model.generate=gen;row={'message':[],'task':'a'};ref={'tokens':[6,9],'eos_ids':[9],'selected_count':2}
  with self.assertRaisesRegex(ValueError,'processor verifier'):
   worker.execute_pixel(w,p,torch,row,lambda:None,ref,lambda x:None)
  for replacement in ({'tokens':[6,9],'eos_ids':[8],'selected_count':2},{'tokens':[6,9],'eos_ids':[9],'selected_count':3}):
   with self.assertRaisesRegex(ValueError,'EOS/K'):
    worker.execute_pixel(w,p,torch,row,lambda:None,replacement,lambda x:None,validate_input=lambda x:None)
 def test_pixel_raw_persisted_before_reference_mismatch(self):
  w=Wrapper();p=SimpleNamespace(compute_merged_keep_indices=lambda:[[1]]);old=w.model.generate
  def gen(**kw):p.compute_merged_keep_indices();return old(**kw)
  w.model.generate=gen;events=[]
  with self.assertRaisesRegex(ValueError,'accepted same-arm'):
   worker.execute_pixel(w,p,torch,{'message':[],'task':'a'},lambda:None,{'tokens':[7,9],'eos_ids':[9],'selected_count':1},lambda x:events.append(x),validate_input=lambda x:None)
  self.assertEqual(events,[[[4,5,6,9]]]);self.assertIs(w.model.generate,gen)
 def test_eadp_call_observer_and_return(self):
  w=Wrapper();native=SimpleNamespace(_greed_select_impl=lambda:(torch.zeros((1,256)),torch.tensor([256])))
  def execute(wrapper,pruner,torch,row,mode,protocol,sync):
   wrapper._processor_inputs();pruner._greed_select_impl();wrapper.model.generate(stopping_criteria=lambda *a:None,do_sample=False,max_new_tokens=2048,top_p=.001,top_k=1,temperature=.01,repetition_penalty=1.)
   return {'answer':'x','output_ids':[9],'eos_ids':[9],'boundaries':{'first_criteria_length':1}}
  mod=ModuleType('runner');mod.execute_one=execute;raw=[]
  with patch.dict(sys.modules,{'runner':mod}):
   value=worker.execute_eadp(w,native,torch,{},lambda:None,{'tokens':[9],'eos_ids':[9],'selected_count':256},lambda x:raw.append(x),validate_input=lambda x:None)
  self.assertEqual(value['dynamic_k'],256);self.assertEqual(raw,[[[4,5,6,9]]])
 def test_failure_retains_raw_and_failure(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'journal';job={'arm':'a'}
   def run(phase,emit):emit([[4,9]]);raise ValueError('failed after generation')
   with self.assertRaises(ValueError):worker.run_two(job,run,FakeCuda(),p)
   self.assertEqual([json.loads(x)['event'] for x in p.read_text().splitlines()],['started','raw_generation','failed'])
 def test_allocator_only_and_both_phases(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'journal'
   def run(phase,emit):emit([[9]]);return {'output_ids':[9]}
   worker.run_two({'arm':'x'},run,FakeCuda(),p)
   rows=[json.loads(x) for x in p.read_text().splitlines()];self.assertEqual(len(rows),6)
   self.assertEqual(rows[-1]['memory']['max_allocated_bytes'],30);self.assertIn('not NVML',rows[-1]['memory']['scope'])
 def test_gpu_and_preflight_locked_before_output(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'out'
   for args in ([],['--preflight-only']):
    r=subprocess.run([sys.executable,str(Path(worker.__file__)),*args,'--out',str(p)],capture_output=True,text=True,timeout=60)
    self.assertNotEqual(r.returncode,0);self.assertFalse(p.exists());self.assertIn('locked',r.stderr)
if __name__=='__main__':unittest.main(verbosity=2)
