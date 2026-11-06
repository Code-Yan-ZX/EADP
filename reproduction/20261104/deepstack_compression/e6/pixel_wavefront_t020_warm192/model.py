"""Model setup used only by an independently admitted warm-process worker."""
import hashlib,importlib.util,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];C=B/'e6/pixel_wavefront_t020_model_timing_v2/model_candidate.py'
def load(name,path,source):
 path=Path(path).resolve()
 if name in sys.modules or hashlib.sha256(path.read_bytes()).hexdigest()!=source[str(path)]:raise ValueError('foreign or changed model dependency')
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def prepare_cpu(source):
 candidate=load('_warm192_existing_model_candidate',C,source);prepared=candidate.prepare_cpu()
 import input_preflight
 if Path(input_preflight.__file__).resolve()!=(B/'e4_c_latency/input_preflight.py').resolve() or prepared['torch'].cuda.is_initialized():raise ValueError('input source or CUDA state')
 e1=B/'e1';rows,refs=input_preflight.load_accepted_e1_references(e1/'e1full001_completion.json',e1/'monitorverifye1full001/started.json',e1/'monitorverifye1full001/terminal.json')
 by={(r['task'],str(r['qid'])):r for r in rows}
 if len(by)!=384 or len(refs)!=384:raise ValueError('accepted E1 identity inventory')
 return candidate,prepared,input_preflight,by,refs

def initialize(candidate,prepared,recipe):
 """Unsafe: caller has already checked independently reviewed GPU admission."""
 import transformers
 from transformers import AutoProcessor
 from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
 from pixelprune import apply_pixelprune
 torch=prepared['torch']
 if torch.cuda.is_initialized() or torch.cuda.get_device_name(0)!='NVIDIA A40' or transformers.__version__!='4.57.1' or getattr(hf.Qwen3VLVisionModel,'__patch_selector_patched__',False):raise ValueError('unplanned model/GPU runtime')
 os.environ.update(PIXELPRUNE_ENABLED='true',PIXELPRUNE_METHOD='pred_2d',PIXELPRUNE_METRIC='max',PIXELPRUNE_THRESHOLD='0.2',PIXELPRUNE_ANCHORED='true',PIXELPRUNE_VIT_LAYER='0',PIXELPRUNE_VERBOSE='false');os.environ.pop('PIXELPRUNE_LOG_FILE',None)
 apply_pixelprune(model='qwen3_vl');patch=prepared['pixel']
 if hf.Qwen3VLVisionModel.forward is not patch._vision_forward or hf.Qwen3VLForConditionalGeneration.forward is not patch._cg_forward:raise ValueError('patch not installed before load')
 start=time.monotonic_ns();processor=AutoProcessor.from_pretrained(candidate.MODEL,local_files_only=True,trust_remote_code=False);model=prepared['model_class'].from_pretrained(candidate.MODEL,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to('cuda').eval();torch.cuda.synchronize();end=time.monotonic_ns()
 if model.dtype!=torch.bfloat16 or model.config._attn_implementation!='flash_attention_2':raise ValueError('wrong actual backend')
 wrapper=object.__new__(prepared['wrapper_class']);wrapper.system_prompt=None;wrapper.min_pixels=wrapper.max_pixels=1048576;wrapper.total_pixels=None;wrapper.fps=None;wrapper.processor=processor;wrapper.model=model;wrapper.verbose=False;wrapper.post_process=False;wrapper.generate_kwargs={k:v for k,v in recipe.items() if k not in ('do_sample','num_beams','use_cache')}
 return wrapper,{'load_start_monotonic_ns':start,'load_end_monotonic_ns':end,'backend':'flash_attention_2','dtype':'bfloat16','GPU':'NVIDIA A40','model_load_outside_TTFT':True,'patch_before_model_load':True}
if __name__=='__main__':raise SystemExit('GPU locked: no reviewed controller, plan or admission')
