"""Actual 384-image 512 processor and independent mRoPE, no model/GPU."""
import argparse,inspect,json,os,socket,sys,time
from pathlib import Path
from types import SimpleNamespace
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2];AUTHOR=ROOT/'_reproduction/author_eadp/Qwen_vl';P=B/'e6/rres_k256_pilot_v2';MODEL=Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve()
 if out!=(D/'run_preflight001/probe').resolve() or not out.is_dir() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('only actual noGPU preflight')
 import preflight_gate,identity_wait;approval=preflight_gate.admit_preflight();identity,launch=identity_wait.wait(preflight_gate.read,(D/'run_preflight001/container_identity.json',D/'run_preflight001/launch.json'))
 cid=identity['container_id']
 if len(cid)!=64 or socket.gethostname()!=cid[:12] or cid not in Path('/proc/self/cgroup').read_text() or launch.get('approval')!=approval:raise ValueError('noGPU container identity')
 sys.path[:0]=[str(D),str(P),str(B/'e4_c_latency'),str(AUTHOR),str(AUTHOR/'VLMEvalKit')]
 import torch,probe_core,resolution,vlmeval.vlm.qwen3_vl.model_fixed_res as module
 from transformers import AutoProcessor,AutoConfig
 from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
 from input_preflight import load_accepted_e1_references,validate_processor
 if Path(inspect.getfile(module.Qwen3VLChatFixedRes)).resolve()!=(AUTHOR/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve() or Path(resolution.__file__).resolve()!=P/'resolution.py' or torch.cuda.is_initialized() or module.QWEN3_FIXED_RESOLUTION!=1024:raise ValueError('actual import/source or CUDA')
 e1=B/'e1';rows,refs=load_accepted_e1_references(e1/'e1full001_completion.json',e1/'monitorverifye1full001/started.json',e1/'monitorverifye1full001/terminal.json')
 anchors=preflight_gate.read(D/'design001.json')['anchors'];processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False);config=AutoConfig.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
 wrapper=object.__new__(module.Qwen3VLChatFixedRes);wrapper.system_prompt=None;wrapper.min_pixels=wrapper.max_pixels=1048576;wrapper.total_pixels=None;wrapper.fps=None;wrapper.processor=processor;wrapper.model=SimpleNamespace(device='cpu',dtype=torch.bfloat16)
 def rope(low):return Qwen3VLModel.get_rope_index(SimpleNamespace(config=config),input_ids=low['input_ids'],image_grid_thw=low['image_grid_thw'],attention_mask=low['attention_mask'])
 probe_core.run(rows,anchors,refs,torch,wrapper,module,config,resolution.fixed_resolution,validate_processor,rope,out)
 if torch.cuda.is_initialized():raise ValueError('CUDA unexpectedly initialized')
if __name__=='__main__':main()
