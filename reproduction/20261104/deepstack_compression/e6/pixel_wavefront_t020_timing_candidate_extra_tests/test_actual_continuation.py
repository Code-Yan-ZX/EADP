"""Pure fake model exercise of the real candidate function; never touches GPU."""
import hashlib,importlib.util,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
B=Path(__file__).resolve().parents[2];P=B/'e6/pixel_wavefront_t020_model_timing_v2/model_candidate.py';sp=importlib.util.spec_from_file_location('_candidate_extra',P);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
class Tests(unittest.TestCase):
 def test_full_continuation_and_bad_image_refused(self):
  with tempfile.TemporaryDirectory() as td:
   image=Path(td)/'image';image.write_bytes(b'image');digest=hashlib.sha256(b'image').hexdigest();job=dict(task='t',qid='1',group_id='g',image_sha256=digest,message_sha256='m',arm='wavefront');row=dict(job,image_path=str(image));ref={'processor':{'i':1},'selected_indices':[4],'tokens':[9],'eos_ids':[9],'generation_kwargs':{'do_sample':False},'selected_count':1};calls=[]
   fake_torch=types.ModuleType('torch');fake_torch.bfloat16='bf16';fake_torch.cuda=types.SimpleNamespace(get_device_name=lambda i:'NVIDIA A40',synchronize=lambda:None)
   transform=types.ModuleType('transformers');transform.__version__='4.57.1';transform.AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *a,**k:'processor')
   models=types.ModuleType('transformers.models');qwen=types.ModuleType('transformers.models.qwen3_vl');hf=types.SimpleNamespace(Qwen3VLVisionModel=type('Vision',(),{}),Qwen3VLForConditionalGeneration=type('Head',(),{}));qwen.modeling_qwen3_vl=hf
   patch_obj=types.SimpleNamespace(_vision_forward=object(),_cg_forward=object());pixel=types.ModuleType('pixelprune')
   def apply_pixelprune(**kw):hf.Qwen3VLVisionModel.forward=patch_obj._vision_forward;hf.Qwen3VLForConditionalGeneration.forward=patch_obj._cg_forward
   pixel.apply_pixelprune=apply_pixelprune;inputs=types.ModuleType('input_preflight');inputs.validate_processor=lambda *args:{'processor':{'i':1}}
   class Model:
    dtype='bf16';config=types.SimpleNamespace(_attn_implementation='flash_attention_2',image_token_id=5)
    def to(self,x):calls.append('model_to');return self
    def eval(self):return self
   model_class=types.SimpleNamespace(from_pretrained=lambda *a,**k:Model());base=types.SimpleNamespace(RECIPE={'do_sample':False,'num_beams':1,'use_cache':True,'top_p':.001})
   def execute(base,selector,candidate,arm,wrapper,patch,torch,row,sync,ref,emit,validate):
    validate('actual');emit([1,9]);return {'tokens':[9],'recipe':ref['generation_kwargs'],'selected_indices':[4],'boundaries':{'t0_before_image_read_ns':10,'first_token_selected_ns':20,'generation_complete_ns':30,'t0_unix_ns':100,'completion_unix_ns':120}}
   prepared={'references':{('t','1','wavefront'):ref},'pixel':patch_obj,'model_class':model_class,'wrapper_class':type('Wrapper',(),{}),'base':base,'selector':object,'candidate':types.SimpleNamespace(select=lambda *a:None),'continuation':types.SimpleNamespace(execute=execute)}
   modules={'torch':fake_torch,'transformers':transform,'transformers.models':models,'transformers.models.qwen3_vl':qwen,'pixelprune':pixel,'input_preflight':inputs}
   with patch.dict(sys.modules,modules):
    raw=[];out=m.unsafe_continuation(prepared,job,row,{'processor':{'i':1},'image_start':2},raw.append);self.assertEqual(out['processor'],{'i':1});self.assertEqual((out['start_monotonic_ns'],out['first_token_monotonic_ns'],out['completion_monotonic_ns']),(10,20,30));self.assertEqual(raw,[[1,9]])
    image.write_bytes(b'tampered');calls.clear()
    with self.assertRaisesRegex(ValueError,'image bytes'):m.unsafe_continuation(prepared,job,row,{'processor':{'i':1},'image_start':2},raw.append)
    self.assertEqual(calls,[])
if __name__=='__main__':unittest.main()
