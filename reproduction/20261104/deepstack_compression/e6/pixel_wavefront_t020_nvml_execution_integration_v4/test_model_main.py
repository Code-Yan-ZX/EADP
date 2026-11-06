"""Exercise real three-arm model continuation with fake CPU-only dependencies."""
import contextlib,importlib.util,os,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
B=Path(__file__).resolve().parents[2];C=B/'e6/pixel_wavefront_t020_nvml_model_candidate'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
model_run=load('_nvml_model_run_test',C/'model_run.py');t0_trace=load('_nvml_t0_trace_test',C/'t0_trace.py')
class FakeTensor:
 def detach(self):return self
 def clone(self):return FakeTensor()
 def equal(self,x):return isinstance(x,FakeTensor)
 def cpu(self):return self
 def tolist(self):return [2,5]
class Selector:
 method='max';threshold=0;anchored=True
 def _select_2d_loco_fast(self,*a):return FakeTensor()
class Cuda:
 def is_initialized(self):return False
 def synchronize(self):pass
 def reset_peak_memory_stats(self):pass
 def max_memory_allocated(self):return 10
 def max_memory_reserved(self):return 20
class Model:
 dtype='bf16';config=types.SimpleNamespace(_attn_implementation='flash_attention_2',image_token_id=42)
 @classmethod
 def from_pretrained(cls,*a,**k):return cls()
 def to(self,x):return self
 def eval(self):return self
class Vision:pass
class Wrapper:pass
class Processor:
 @staticmethod
 def from_pretrained(*a,**k):return object()
class Continuation:
 @staticmethod
 @contextlib.contextmanager
 def timed_measure(base):yield
 @staticmethod
 def execute(base,selector,candidate,arm,wrapper,patch,torch,row,sync,ref,emit,validate):
  if arm not in ('author','wavefront'):raise ValueError('wrong arm')
  validate({});emit([[1,9]]);return {'selected_indices':[2,5],'tokens':[9],'eos_ids':[9],'recipe':{'x':1}}
class ModelMainTest(unittest.TestCase):
 def test_three_real_branches_and_same_arm_identity(self):
  expected={'processor':{'input_ids':{'shape':[1,1]}},'selected_indices':[2,5],'tokens':[9],'eos_ids':[9]};torch=types.SimpleNamespace(cuda=Cuda(),bfloat16='bf16');hf=types.ModuleType('transformers.models.qwen3_vl.modeling_qwen3_vl');hf.Qwen3VLVisionModel=Vision;hf.Qwen3VLForConditionalGeneration=Model
  tr=types.ModuleType('transformers');tr.__version__='4.57.1';tr.AutoProcessor=Processor;tm=types.ModuleType('transformers.models');tq=types.ModuleType('transformers.models.qwen3_vl');tq.modeling_qwen3_vl=hf
  pixel=types.SimpleNamespace(_vision_forward=lambda *a:None,_cg_forward=lambda *a:None)
  pix=types.ModuleType('pixelprune')
  def apply_pixelprune(**kw):Vision.forward=pixel._vision_forward;Model.forward=pixel._cg_forward
  pix.apply_pixelprune=apply_pixelprune
  inp=types.ModuleType('input_preflight');inp.validate_processor=lambda *a:{'processor':expected['processor']}
  def execute_pixel(wrapper,patch,torch,row,sync,ref,emit,observe_count,validate_input):
   self.assertEqual(os.environ['PIXELPRUNE_THRESHOLD'],'0.0');Selector()._select_2d_loco_fast(None,1,1,None);validate_input({});emit([[1,9]]);return {'tokens':[9],'eos_ids':[9],'recipe':{'x':1}}
  base=types.SimpleNamespace(RECIPE={'do_sample':False,'num_beams':1,'use_cache':True,'max_new_tokens':1},execute_pixel=execute_pixel);prepared={'torch':torch,'base':base,'selector':Selector,'model_path':'fake','model_class':Model,'wrapper_class':Wrapper,'pixel':pixel,'candidate':types.SimpleNamespace(select=lambda:None),'continuation':Continuation};row={'task':'fake','qid':'1','group_id':'g','image_sha256':'sha','message_sha256':'m','image_path':'fake'};e1={'processor':expected['processor'],'image_start':1};mods={'transformers':tr,'transformers.models':tm,'transformers.models.qwen3_vl':tq,'transformers.models.qwen3_vl.modeling_qwen3_vl':hf,'pixelprune':pix,'input_preflight':inp}
  with patch.dict(sys.modules,mods),patch.object(model_run,'sha',return_value='sha'),patch.object(model_run,'load',return_value=t0_trace):
   for arm in ('t0_author','t020_author','t020_wavefront'):
    events=[];raw=[];job=dict(row,arm=arm);model,wrapper,result=model_run.execute(prepared,job,row,e1,expected,events.append,raw.append)
    self.assertEqual(result['selected_indices'],[2,5]);self.assertEqual(raw,[[[1,9]]]);self.assertEqual([e['event'] for e in events],['model_load_start','model_load_end','request_start','request_end']);self.assertEqual(os.environ['PIXELPRUNE_THRESHOLD'],'0.0' if arm=='t0_author' else '0.2')
if __name__=='__main__':unittest.main()
