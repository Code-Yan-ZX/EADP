"""Actual no-GPU dependency and processor preflight; no model construction."""
import hashlib,inspect,json,os,sys,importlib.util
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];ROOT=BASE.parents[2]
E4=BASE/'e4_c_latency';PIXEL=BASE/'e6/external_sources/pixelprune_8d347eb';AUTHOR=ROOT/'_reproduction/author_eadp/Qwen_vl/VLMEvalKit'
sys.path[:0]=[str(HERE),str(E4),str(AUTHOR),str(PIXEL)]

def _load_bootstrap():
 spec=importlib.util.spec_from_file_location('_pixel_pilot_bootstrap_v2',HERE/'bootstrap.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.pin_all();return module
_bootstrap=_load_bootstrap()
_bootstrap.pin('worker',HERE/'worker.py')
from pilot_contract import validate_schedule

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def verify_source_plan(plan):
 for path,digest in plan['sha256'].items():
  if sha(path)!=digest:raise ValueError('frozen dependency mismatch: '+path)
 validate_schedule(plan['jobs'],plan['anchors'])

def validate_upstream():
 checks=[('e6/pixelprune_dev384_accept/verification001.json','e1_independent_review/pixel_dev384_result_review.json','e6/pixelprune_dev384_accept/monitor_verify001','e1_independent_review/monitor_pixel_dev384_result'),('e6/pixelprune_t045_accept/verification001.json','e1_independent_review/pixel_t045_dev384_result_review.json','e6/pixelprune_t045_accept/monitor_verify001','e1_independent_review/monitor_pixel_t045_dev384_result')]
 for formal,review,formalmon,reviewmon in checks:
  f=read(BASE/formal);r=read(BASE/review)
  if f.get('status')!='passed' or f.get('questions')!=384 or f.get('errors')!=[] or r.get('status')!='passed' or r.get('independent_review') is not True or r.get('questions')!=384 or r.get('errors')!=[] or r.get('formal_sha256')!=sha(BASE/formal):raise ValueError('upstream pixel acceptance failed')
  for mon in (formalmon,reviewmon):
   start=read(BASE/mon/'started.json');term=read(BASE/mon/'terminal.json')
   if term.get('reason')!='completed' or term.get('worker_returncode')!=0 or term.get('remaining_process_group')!=[] or term.get('worker_pid')!=start.get('worker_pid'):raise ValueError('upstream pixel supervisor failed')
  if r.get('formal_started_sha256')!=sha(BASE/formalmon/'started.json') or r.get('formal_terminal_sha256')!=sha(BASE/formalmon/'terminal.json'):raise ValueError('upstream pixel proof binding')
 return True

def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
 plan=read(HERE/'plan001.json');verify_source_plan(plan);validate_upstream()
 import torch,transformers
 import input_preflight,worker,observer
 gpu_launch=_bootstrap.pin('gpu_launch_v4',HERE/'gpu_launch.py')
 gpu_worker=_bootstrap.pin('gpu_worker_candidate_v4',HERE/'gpu_worker_candidate.py')
 import pixelprune.patches.qwen3_vl_hf as pix
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes
 from transformers import AutoProcessor,AutoConfig
 if torch.cuda.is_initialized():raise ValueError('CUDA initialized before no-GPU preflight')
 for mod,path in [(input_preflight,E4/'input_preflight.py'),(worker,HERE/'worker.py'),(observer,HERE/'observer.py'),(gpu_launch,HERE/'gpu_launch.py'),(gpu_worker,HERE/'gpu_worker_candidate.py'),(pix,PIXEL/'pixelprune/patches/qwen3_vl_hf.py')]:
  if Path(mod.__file__).resolve()!=path.resolve():raise ValueError('foreign module: '+str(path))
 if Path(inspect.getfile(Qwen3VLChatFixedRes)).resolve()!=(AUTHOR/'vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve():raise ValueError('wrong wrapper')
 if transformers.__version__!='4.57.1':raise ValueError('unexpected transformers')
 reference_module=_bootstrap.pin('reference_candidate',HERE/'reference_candidate.py')
 same_arm=reference_module.load_refs(plan['anchors'])
 if len(same_arm)!=9:raise ValueError('missing same-arm output references')
 e1=BASE/'e1';rows,refs=input_preflight.load_accepted_e1_references(e1/'e1full001_completion.json',e1/'monitorverifye1full001/started.json',e1/'monitorverifye1full001/terminal.json')
 lookup={(r['task'],str(r['qid'])):r for r in rows}
 if len(lookup)!=384:raise ValueError('expected complete DEV input set')
 weights=read(ROOT/'reproduction/20261028/qwen/qwen_weight_identity.json')['files'];stats=[]
 if len(weights)!=4:raise ValueError('four weight shards required')
 for item in weights:
  p=Path(item['path']);s=p.stat()
  before=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
  if s.st_size!=item['bytes'] or sha(p)!=item['expected_lfs_sha256']:raise ValueError('weight hash mismatch')
  s=p.stat();after=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
  if before!=after:raise ValueError('weight changed while hashing')
  stats.append({'path':str(p),'stat':after,'sha256':item['expected_lfs_sha256']})
 model=Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
 processor=AutoProcessor.from_pretrained(model,local_files_only=True,trust_remote_code=False);config=AutoConfig.from_pretrained(model,local_files_only=True,trust_remote_code=False)
 stub=object.__new__(Qwen3VLChatFixedRes);stub.system_prompt=None;stub.min_pixels=stub.max_pixels=1048576;stub.total_pixels=None;stub.processor=processor;stub.model=SimpleNamespace(device='cpu',dtype=torch.bfloat16)
 checked=[]
 for anchor in plan['anchors']:
  key=(anchor['task'],str(anchor['qid']));row=lookup[key];ref=refs[key]
  if row.get('split')!='DEV' or any(row.get(k)!=anchor.get(k) for k in ('task','group_id','message_sha256','image_sha256')) or sha(row['image_path'])!=anchor['image_sha256']:raise ValueError('anchor identity mismatch')
  actual=stub._processor_inputs(stub._build_messages(row['message'],dataset=row['task']))
  input_preflight.validate_processor(torch,actual,ref['processor'],config.image_token_id,ref['image_start'])
  checked.append({'task':key[0],'qid':key[1],'group_id':row['group_id'],'image_start':ref['image_start']})
  del actual
 if torch.cuda.is_initialized():raise ValueError('unexpected GPU initialization')
 out={'status':'passed','stage':'actual no-GPU preflight; no model or execution admission','source_plan_sha256':sha(HERE/'plan001.json'),
      'cuda_initialized':False,'model_loaded':False,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,
      'prediction_scope':'accepted E1 DEV mixed journal and Pixel DEV records read for same-arm input/output/K/EOS identity', 'same_arm_references':len(same_arm),
      'torch':torch.__version__,'transformers':transformers.__version__,'weight_sha256_verified':4,'weight_stats':stats,'anchors':checked,'jobs':plan['jobs']}
 with args.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
