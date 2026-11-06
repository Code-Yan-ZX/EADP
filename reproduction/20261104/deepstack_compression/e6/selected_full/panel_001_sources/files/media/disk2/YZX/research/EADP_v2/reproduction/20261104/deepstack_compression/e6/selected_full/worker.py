"""Identical no-GPU preflight and bounded Div/CD DEV panel preparation."""
import argparse,importlib.metadata,inspect,json,os,sys,time,traceback
from pathlib import Path
from full_imports import HERE,SMOKE,load,prepare
ROOT=HERE.parents[1];E4=ROOT/'e4'
_plan=load('eadp_selected_full_plan',HERE/'freeze_plan.py');sha=_plan.sha;build=_plan.build
def event(path,record):
 with path.open('a') as f:f.write(json.dumps(record,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--preflight-only',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if not a.execute:raise ValueError('explicit execute required')
 gate=load('eadp_selected_full_gate',HERE/'gate.py');approval=gate.check(preflight_only=a.preflight_only)
 out=a.out.resolve();launch=json.loads((out/'launch.json').read_text());receipt=out/'host_receipt.json'
 if launch.get('host_receipt')!={'path':str(receipt),'sha256':sha(receipt)} or json.loads(receipt.read_text())!=approval:
  raise ValueError('host receipt differs')
 plan=json.loads((HERE/'plan.json').read_text())
 if plan!=build():raise ValueError('DEV plan changed')
 prepare();iw=load('identity_worker',E4/'identity_worker.py');author=json.loads(iw.PLAN.read_text())
 for group in ('source_sha256','implementation_sha256'):
  for path,digest in author[group].items():
   if sha(path)!=digest:raise ValueError('author implementation changed')
 admission=load('eadp_full_e1_admission',ROOT/'m3_oracle_exec/e1_admission.py');e1=admission.validate(metadata_only=a.preflight_only)
 rows=[json.loads(s) for s in Path(plan['inputs_path']).read_text().splitlines()]
 if len(rows)!=384 or any(sha(row['image_path'])!=row['image_sha256'] for row in rows):raise ValueError('DEV image inventory mismatch')
 sys.path[:0]=[str(HERE),str(E4),author['author_root'],str(Path(author['author_root'])/'VLMEvalKit')]
 import torch
 from transformers import AutoTokenizer
 from model.cdpruner import CDPruner
 from model.divpruner import DivPruner
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatCDPruner,Qwen3VLChatDivPruner
 root=Path(author['author_root']).resolve();wrapper=root/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py'
 classes={CDPruner:root/'model/cdpruner.py',DivPruner:root/'model/divpruner.py',Qwen3VLChatCDPruner:wrapper,Qwen3VLChatDivPruner:wrapper}
 if any(Path(inspect.getfile(c)).resolve()!=p.resolve() for c,p in classes.items()):raise ValueError('author class path mismatch')
 versions={k:importlib.metadata.version(k) for k in author['expected_versions']}
 if versions!=author['expected_versions']:raise ValueError('package versions differ')
 runner=load('eadp_selected_full_runner',HERE/'full_runner.py');tokenizer=AutoTokenizer.from_pretrained(author['recipe']['model_path'],local_files_only=True,trust_remote_code=False)
 if a.preflight_only:
  if torch.cuda.is_initialized():raise ValueError('preflight initialized CUDA')
  print(json.dumps({'status':'passed','cuda_initialized':False,'model_loaded':False,'GT_opened':False,'predictions_opened':False,
      'inputs_verified':len(rows),'weights_sha256_verified':4,'source_count_verified':approval['source_count_verified'],
      'source_manifest_sha256':approval['source_manifest_sha256'],'versions':versions,'classes':{c.__name__:str(p) for c,p in classes.items()}}),flush=True);return
 if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
 expected_controls=load('eadp_selected_full_generation_reference',HERE/'generation_reference.py').controls()
 if any((out/x).exists() for x in ('frozen.json','records.jsonl','generated.jsonl','events.jsonl','worker_complete.json')):raise ValueError('reused output')
 recipe=dict(author['recipe']);recipe.pop('alpha');recipe.pop('beta');recipe['visual_token_num']=256
 weights=json.loads(iw.WEIGHTS.read_text())['files'];stats={}
 for w in weights:
  s=Path(w['path']).stat();stats[w['path']]=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
 with (out/'frozen.json').open('x') as f:
  json.dump({'stage':'full_dev','approval':approval,'plan_sha256':sha(HERE/'plan.json'),'e1_completion_sha256':e1.proof_sha,
      'weights_identity_sha256':sha(iw.WEIGHTS),'weight_stats_before':stats,'questions':384,'generations':1536,
      'GT_opened':False,'scoring':False,'full_tensors_archived':False},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 t=time.monotonic();event(out/'events.jsonl',{'event':'worker_started','unix':time.time()})
 try:
  import gc
  for method,cls,pruner in (('div',Qwen3VLChatDivPruner,DivPruner),('cd',Qwen3VLChatCDPruner,CDPruner)):
   w=cls(**recipe)
   if w.model.dtype!=torch.bfloat16 or w.model.config._attn_implementation!='flash_attention_2':raise ValueError('model backend mismatch')
   event(out/'events.jsonl',{'event':'model_loaded','method':method,'unix':time.time()});runner.run_method(torch,w,cls,pruner,method,rows,recipe,out,expected_controls[method])
   event(out/'events.jsonl',{'event':'method_completed','method':method,'unix':time.time()});del w;gc.collect();torch.cuda.empty_cache()
  for path,before in stats.items():
   s=Path(path).stat()
   if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=before:raise ValueError('model weights changed')
  gate.check()
  with (out/'worker_complete.json').open('x') as f:json.dump({'status':'completed','questions':384,'generations':1536,'pairs':768,
       'GT_opened':False,'scoring':False,'seconds':time.monotonic()-t},f);f.write('\n');f.flush();os.fsync(f.fileno())
  event(out/'events.jsonl',{'event':'worker_completed','unix':time.time()})
 except BaseException as exc:
  with (out/'failure.json').open('x') as f:
   json.dump({'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),
              'oom':isinstance(exc,torch.cuda.OutOfMemoryError),'seconds':time.monotonic()-t},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  event(out/'events.jsonl',{'event':'worker_failed','type':type(exc).__name__,'unix':time.time()});raise
if __name__=='__main__':main()
