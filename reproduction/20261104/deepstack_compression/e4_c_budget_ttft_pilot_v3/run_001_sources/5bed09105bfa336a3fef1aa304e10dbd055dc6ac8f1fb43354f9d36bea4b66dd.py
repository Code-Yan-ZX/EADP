"""Prepared single-job model continuation; entry remains unconditionally GPU-locked.

A future reviewed controller must verify all source/endpoint/reference and
supervisor gates before calling continue_after_admission. This file itself does
not grant or claim such admission.
"""
import importlib.util,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];ROOT=BASE.parents[2]
spec=importlib.util.spec_from_file_location('_pixel_bootstrap_model_v3',HERE/'bootstrap.py');boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.pin_all()
pre=boot.pin('preflight',HERE/'preflight.py');worker=boot.pin('worker',HERE/'worker.py');refs_module=boot.pin('reference_candidate',HERE/'reference_candidate.py')

MODEL=Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
AUTHOR=ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit';PIXEL=BASE/'e6/external_sources/pixelprune_8d347eb'

def continue_after_admission(job,out):
 """Unsafe continuation: not an admission check; do not call directly."""
 import torch,transformers
 from input_preflight import load_accepted_e1_references,validate_processor
 plan=pre.read(HERE/'plan001.json');pre.verify_source_plan(plan)
 if job not in plan['jobs']:raise ValueError('job not in frozen schedule')
 refs=refs_module.load_refs(plan['anchors']);key=(job['arm'],job['task'],str(job['qid']));ref=refs[key]
 rows,e1=load_accepted_e1_references(BASE/'e1/e1full001_completion.json',BASE/'e1/monitorverifye1full001/started.json',BASE/'e1/monitorverifye1full001/terminal.json')
 selected=[x for x in rows if (x['task'],str(x['qid']))==(job['task'],str(job['qid']))]
 if len(selected)!=1:raise ValueError('missing frozen row')
 row=selected[0];er=e1[(row['task'],str(row['qid']))]
 if row['group_id']!=job['group_id'] or er['processor']!=ref['processor'] or pre.sha(row['image_path'])!=row['image_sha256']:raise ValueError('input reference mismatch')
 if transformers.__version__!='4.57.1' or torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('runtime backend mismatch')
 sys.path[:0]=[str(ROOT/'_reproduction/author_eadp/Qwen_vl'),str(AUTHOR),str(PIXEL)]
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes,Qwen3VLChatEADP
 out=Path(out);out.mkdir(exist_ok=False);journal=out/'journal.jsonl'
 worker.append(journal,{'event':'model_loading','job':job})
 begin=time.perf_counter_ns()
 if job['arm']=='eadp_k256':
  from model import pruner as native
  if Path(native.__file__).resolve()!=(ROOT/'_reproduction/author_eadp/Qwen_vl/model/pruner.py').resolve():raise ValueError('foreign EADP pruner')
  author=pre.read(ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json');wrapper=Qwen3VLChatEADP(**author['recipe']);pixel=None
 else:
  from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
  from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
  from pixelprune import apply_pixelprune
  import pixelprune.patches.qwen3_vl_hf as pixel
  if Path(pixel.__file__).resolve()!=(PIXEL/'pixelprune/patches/qwen3_vl_hf.py').resolve():raise ValueError('foreign pixel patch')
  if getattr(hf.Qwen3VLVisionModel,'__patch_selector_patched__',False):raise ValueError('pre-existing pixel patch')
  os.environ.update(PIXELPRUNE_ENABLED='true',PIXELPRUNE_METHOD='pred_2d',PIXELPRUNE_METRIC='max',PIXELPRUNE_THRESHOLD='0.0' if job['arm']=='pixel_t0' else '0.45',PIXELPRUNE_ANCHORED='true',PIXELPRUNE_VIT_LAYER='0',PIXELPRUNE_VERBOSE='false')
  os.environ.pop('PIXELPRUNE_LOG_FILE',None);apply_pixelprune(model='qwen3_vl')
  if hf.Qwen3VLVisionModel.forward is not pixel._vision_forward or hf.Qwen3VLForConditionalGeneration.forward is not pixel._cg_forward:raise ValueError('wrong patch dispatch')
  processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
  model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to('cuda').eval()
  wrapper=object.__new__(Qwen3VLChatFixedRes);wrapper.system_prompt=None;wrapper.min_pixels=wrapper.max_pixels=1048576;wrapper.total_pixels=None;wrapper.fps=None;wrapper.processor=processor;wrapper.model=model;wrapper.verbose=False;wrapper.post_process=False
  wrapper.generate_kwargs={k:v for k,v in worker.RECIPE.items() if k not in ('do_sample','num_beams','use_cache')}
  native=None
 if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('actual model backend mismatch')
 torch.cuda.synchronize();setup_end=time.perf_counter_ns()
 row['image_start']=er['image_start']
 worker.append(journal,{'event':'model_ready','job':job,'setup_wall_ns':setup_end-begin,'allocator_post_load':worker.memory_snapshot(torch.cuda)})
 def verify_actual(inputs):
  validate_processor(torch,inputs,ref['processor'],wrapper.model.config.image_token_id,er['image_start'])
 def run(phase,emit):
  observed=phase=='measured'
  if native is None:return worker.execute_pixel(wrapper,pixel,torch,row,torch.cuda.synchronize,ref,emit,observe_count=observed,validate_input=verify_actual)
  return worker.execute_eadp(wrapper,native,torch,row,torch.cuda.synchronize,ref,emit,observe_count=observed,validate_input=verify_actual)
 worker.run_two(job,run,torch.cuda,journal)
 worker.append(journal,{'event':'job_complete','job':job})

def validate_release(job,launch,release,hostname,launch_sha,plan_sha):
 cid=release.get('container_id');host=release.get('host_init') or {}
 if (not isinstance(cid,str) or len(cid)!=64 or hostname!=cid[:12] or release.get('job')!=job or
     release.get('plan_sha256')!=plan_sha or release.get('launch_sha256')!=launch_sha or
     release.get('controller')!=launch.get('controller') or type(host.get('pid')) is not int or host['pid']<=0 or
     type(host.get('start_ticks')) is not int or host['start_ticks']<=0 or not host.get('boot_id')):raise ValueError('controller/container release mismatch')
 return True

def wait_release(job,jobdir,launch,plan_sha,timeout=60):
 import socket
 begin=time.monotonic();path=jobdir/'release.json'
 while not path.exists():
  if time.monotonic()-begin>=timeout:raise ValueError('controller release timed out')
  time.sleep(.2)
 release=pre.read(path);validate_release(job,launch,release,socket.gethostname(),pre.sha(jobdir/'launch.json'),plan_sha)
 return release

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--job-index',type=int,required=True);p.add_argument('--handshake-only',action='store_true');a=p.parse_args()
 if a.handshake_only:
  import torch
  if a.job_index!=0 or torch.cuda.is_initialized():raise ValueError('invalid no-GPU handshake state')
  plan=pre.read(HERE/'plan001.json');pre.verify_source_plan(plan);job=plan['jobs'][0];jobdir=HERE/'run_handshake001/job_000';launch=pre.read(jobdir/'launch.json')
  if launch.get('plan_sha256')!=pre.sha(HERE/'plan001.json') or launch.get('job')!=job:raise ValueError('handshake launch binding')
  release=wait_release(job,jobdir,launch,pre.sha(HERE/'plan001.json'))
  if torch.cuda.is_initialized():raise ValueError('CUDA initialized during no-GPU handshake')
  result={'status':'passed','cuda_initialized':False,'model_loaded':False,'GT_opened':False,'scoring_performed':False,'job':job,'container_id':release['container_id'],'launch_sha256':pre.sha(jobdir/'launch.json'),'release_sha256':pre.sha(jobdir/'release.json')}
  with (jobdir/'handshake.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  return 0
 admission=boot.pin('pilot_admission_v4',HERE/'admission.py')
 try:admitted=admission.require()
 except Exception as exc:raise ValueError('GPU locked: independent admission required') from exc
 plan=pre.read(HERE/'plan001.json');jobs=plan['jobs']
 if a.job_index not in range(len(jobs)):raise ValueError('invalid frozen job index')
 job=jobs[a.job_index];jobdir=HERE/'run_pilot001'/('job_%03d'%a.job_index);launch=pre.read(jobdir/'launch.json')
 if launch.get('plan_sha256')!=admitted['plan_sha256'] or launch.get('job')!=job:raise ValueError('launcher job binding mismatch')
 cmd=launch.get('command',[])
 if cmd.count(str(HERE/'gpu_worker_candidate.py'))!=1 or cmd.count('--job-index')!=1 or cmd[cmd.index('--job-index')+1]!=str(a.job_index):raise ValueError('worker launch argv mismatch')
 wait_release(job,jobdir,launch,admitted['plan_sha256'])
 try:return continue_after_admission(job,jobdir/'observations')
 except BaseException as exc:
  import traceback
  worker.append(jobdir/'worker_failure.jsonl',{'job':job,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
  raise
if __name__=='__main__':main()
