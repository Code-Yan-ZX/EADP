"""Same worker for real no-GPU preflight and locked future real-model arms."""
import argparse,importlib.metadata,inspect,json,os,sys,time,traceback
from pathlib import Path
from local_imports import HERE,ROOT,E4,BASE,pinned,prepare
from freeze_plan import sha,plan as rebuild_plan

def append(path,obj):
 with Path(path).open('a') as f:f.write(json.dumps(obj,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--preflight-only',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if not a.execute:raise ValueError('explicit execution required')
 gate=pinned('eadp_selected_gate',HERE/'gate.py');approval=gate.check(preflight_only=a.preflight_only)
 out=a.out.resolve();launch=json.loads((out/'launch.json').read_text());receipt=out/'host_receipt.json'
 if launch.get('host_receipt')!={'path':str(receipt),'sha256':sha(receipt)} or json.loads(receipt.read_text())!=approval:
  raise ValueError('host admission receipt mismatch')
 frozen=json.loads((HERE/'plan.json').read_text())
 if frozen!=rebuild_plan():raise ValueError('frozen plan, inputs, native anchors or budget changed')
 sys.path.insert(0,str(E4));identity=pinned('identity_worker',E4/'identity_worker.py')
 author=json.loads(identity.PLAN.read_text())
 for group in ('source_sha256','implementation_sha256'):
  for path,digest in author[group].items():
   if sha(path)!=digest:raise ValueError('author implementation changed: '+path)
 admission=pinned('eadp_selected_e1_admission',ROOT/'m3_oracle_exec/e1_admission.py')
 e1=admission.validate(metadata_only=a.preflight_only)
 completion=json.loads((ROOT/'e1/e1full001_completion.json').read_text());ip=Path(completion['files']['inputs']['path'])
 allrows=[json.loads(s) for s in ip.read_text().splitlines()];rows=[]
 for anchor in frozen['anchors']:
  found=[r for r in allrows if r['task']==anchor['task'] and str(r['qid'])==str(anchor['qid'])]
  if len(found)!=1 or any(str(found[0].get(k))!=str(v) for k,v in anchor.items()):raise ValueError('anchor input mismatch')
  rows.append(found[0])
 sys.path[:0]=[str(HERE),str(E4),author['author_root'],str(Path(author['author_root'])/'VLMEvalKit')]
 import torch
 from transformers import AutoTokenizer
 from model.cdpruner import CDPruner
 from model.divpruner import DivPruner
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatCDPruner,Qwen3VLChatDivPruner
 root=Path(author['author_root']).resolve()
 expected={CDPruner:root/'model/cdpruner.py',DivPruner:root/'model/divpruner.py',
           Qwen3VLChatCDPruner:root/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py',
           Qwen3VLChatDivPruner:root/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py'}
 if any(Path(inspect.getfile(cls)).resolve()!=path.resolve() for cls,path in expected.items()):raise ValueError('author class source mismatch')
 versions={k:importlib.metadata.version(k) for k in author['expected_versions']}
 if versions!=author['expected_versions']:raise ValueError('package versions differ')
 stock,contract=prepare();runner=pinned('eadp_selected_arm_runner',HERE/'arm_runner.py')
 tokenizer=AutoTokenizer.from_pretrained(author['recipe']['model_path'],local_files_only=True,trust_remote_code=False)
 if not tokenizer.encode(' the',add_special_tokens=False):raise ValueError('forced token unavailable')
 if a.preflight_only:
  if torch.cuda.is_initialized():raise ValueError('CUDA initialized during preflight')
  print(json.dumps({'status':'passed','cuda_initialized':False,'model_loaded':False,'GT_opened':False,'predictions_opened':False,
                    'anchors_verified':len(rows),'weights_sha256_verified':4,'versions':versions,
                    'source_count_verified':approval['source_count_verified'],'source_manifest_sha256':approval['source_manifest_sha256'],
                    'classes':{c.__name__:str(path) for c,path in expected.items()}}),flush=True);return
 if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
 recipe=dict(author['recipe']);recipe.pop('alpha');recipe.pop('beta');recipe['visual_token_num']=256
 stats={x['path']:[Path(x['path']).stat().st_dev,Path(x['path']).stat().st_ino,Path(x['path']).stat().st_size,Path(x['path']).stat().st_mtime_ns,Path(x['path']).stat().st_ctime_ns]
        for x in json.loads(identity.WEIGHTS.read_text())['files']}
 if any((out/x).exists() for x in ('records.jsonl','generated.jsonl','events.jsonl','worker_complete.json','arrays')):raise ValueError('output path reused')
 with (out/'frozen.json').open('x') as f:
  json.dump({'approval':approval,'plan_sha256':sha(HERE/'plan.json'),'weights_identity_sha256':sha(identity.WEIGHTS),
             'weight_stats_before':stats,'e1_proof_sha256':e1.proof_sha,'anchors':frozen['anchors'],'GT_opened':False,'scoring':False},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 start=time.monotonic();append(out/'events.jsonl',{'event':'worker_started','unix':time.time()})
 try:
  import gc
  for method,cls,pruner in (('div',Qwen3VLChatDivPruner,DivPruner),('cd',Qwen3VLChatCDPruner,CDPruner)):
   t=time.monotonic();torch.cuda.reset_peak_memory_stats();w=cls(**recipe)
   if w.model.dtype!=torch.bfloat16 or w.model.config._attn_implementation!='flash_attention_2':raise ValueError('model backend differs')
   append(out/'events.jsonl',{'event':'model_loaded','method':method,'seconds':time.monotonic()-t,'unix':time.time()})
   runner.run(torch,w,cls,pruner,method,rows,recipe,out)
   append(out/'events.jsonl',{'event':'method_done','method':method,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                              'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'unix':time.time()})
   del w;gc.collect();torch.cuda.empty_cache()
  for path,before in stats.items():
   s=Path(path).stat()
   if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=before:raise ValueError('weights changed')
  gate.check(preflight_only=False)
  with (out/'worker_complete.json').open('x') as f:json.dump({'questions':3,'generations':16,'paired':8,'seconds':time.monotonic()-start,'GT_opened':False},f);f.write('\n');f.flush();os.fsync(f.fileno())
  append(out/'events.jsonl',{'event':'worker_complete','unix':time.time()})
 except BaseException as exc:
  with (out/'failure.json').open('x') as f:
   json.dump({'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'seconds':time.monotonic()-start,
              'oom':isinstance(exc,torch.cuda.OutOfMemoryError)},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  append(out/'events.jsonl',{'event':'worker_failed','unix':time.time(),'type':type(exc).__name__});raise
if __name__=='__main__':main()
