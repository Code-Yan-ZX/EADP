"""Prepared full DEV E2/E3 worker; hard-locked until external approval.

The launch gate currently always rejects. In particular, --preflight-only
is not a bypass. This file contains no GT or scoring code.
"""
import argparse,hashlib,importlib.metadata,inspect,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;SMOKE=BASE/'e2_e3_smoke';E4=BASE/'e4';CONTRACT=BASE/'e2_e3_contract'
from full_imports import local
_gate=local('panel_gate');execution_gate=_gate.execution_gate;smoke_gate=_gate.smoke_gate;b_gate=_gate.b_gate;sha=_gate.sha
_contract=local('contract');bind_dev_inputs=_contract.bind_dev_inputs;inventory=_contract.inventory
run_panel=local('panel_runner').run_panel

def inputs(proof):
 def lines(p):
  with open(p,encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]
 files=proof['files'];rows=bind_dev_inputs(lines(files['inputs']['path']),lines(files['panel']['path']));inventory(rows)
 for row in rows:
  if any(x in row for x in ('answer','answers','prediction','ground_truth')):raise ValueError('DEV input contains answer/prediction')
  messages=row['message'];images=[x['value'] for x in messages if x.get('type')=='image']
  if images!=[row['image_path']] or sha(row['image_path'])!=row['image_sha256'] or hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest()!=row['message_sha256']:
   raise ValueError('prompt or image identity mismatch')
 refs={}
 for rec in lines(files['journal']['path']):
  if rec.get('forced') is False and rec.get('arm') in ('P3-D0','P3-D1'):
   key=(rec['task'],str(rec['qid']),rec['arm'])
   if key in refs:raise ValueError('duplicate E1 endpoint')
   refs[key]=rec
 if len(refs)!=768 or any((r['task'],str(r['qid']),arm) not in refs for r in rows for arm in ('P3-D0','P3-D1')):raise ValueError('E1 endpoint inventory')
 return rows,refs

def runtime_imports(plan):
 sys.path[:0]=[str(SMOKE),str(E4),str(CONTRACT),plan['author_root'],str(Path(plan['author_root'])/'VLMEvalKit')]
 import torch,model_adapter
 from local_imports import load_local
 runner=load_local('intervention_runner')
 from model import pruner
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP,Qwen3VLChatCDPruner
 got={'runner':Path(runner.__file__).resolve(),'adapter':Path(model_adapter.__file__).resolve(),
      'pruner':Path(pruner.__file__).resolve(),'wrapper':Path(inspect.getfile(Qwen3VLChatEADP)).resolve()}
 expected={'runner':SMOKE/'intervention_runner.py','adapter':CONTRACT/'model_adapter.py',
           'pruner':Path(plan['author_root'])/'model/pruner.py',
           'wrapper':Path(plan['author_root'])/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py'}
 if got!={k:p.resolve() for k,p in expected.items()}:raise ImportError('unexpected model import origins')
 versions={name:importlib.metadata.version(name) for name in plan['expected_versions']}
 if versions!=plan['expected_versions']:raise ValueError('runtime version mismatch')
 return torch,runner,pruner,Qwen3VLChatEADP,Qwen3VLChatCDPruner,got,versions

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--preflight-only',action='store_true')
 for k in ('e1-acceptance','e1-started','e1-terminal','out'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args()
 if not a.execute:raise ValueError('explicit execution required')
 approval=execution_gate(preflight=a.preflight_only)
 upstream={'smoke':smoke_gate(),'e4b':b_gate()}
 sys.path.insert(0,str(E4));from identity_worker import PLAN,WEIGHTS,check_e1
 proof=check_e1(a.e1_acceptance,a.e1_started,a.e1_terminal);rows,refs=inputs(proof);plan=json.loads(PLAN.read_text())
 expected={'visual_token_num':256,'alpha':0.5,'beta':2.0,'min_pixels':1048576,'max_pixels':1048576,'max_new_tokens':2048,'use_custom_prompt':True,'use_vllm':False}
 if any(plan['recipe'].get(k)!=v for k,v in expected.items()):raise ValueError('recipe mismatch')
 for group in ('source_sha256','implementation_sha256'):
  for path,digest in plan[group].items():
   if sha(path)!=digest:raise ValueError('author source changed')
 weights=json.loads(WEIGHTS.read_text())['files'];stats={}
 if len(weights)!=4:raise ValueError('weight inventory')
 for item in weights:
  path=Path(item['path']);s=path.stat()
  if s.st_size!=item['bytes'] or sha(path)!=item['expected_lfs_sha256']:raise ValueError('weight mismatch')
  stats[str(path)]=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
 torch,runner,pruner,Wrapper,BaseWrapper,imports,versions=runtime_imports(plan)
 if a.preflight_only:
  if torch.cuda.is_initialized():raise ValueError('preflight initialized CUDA')
  print(json.dumps({'status':'preflight_passed','cuda_initialized':False,'model_loaded':False,'predictions':False,'imports':{k:str(v) for k,v in imports.items()},'versions':versions}),flush=True);return
 if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('wrong GPU')
 out=a.out.resolve()
 if not out.is_dir() or (out/'frozen.json').exists() or (out/'data').exists():raise ValueError('invalid/reused output')
 frozen_smoke=json.loads((SMOKE/'run_001/frozen.json').read_text()); anchor_rows=[]
 for wanted in frozen_smoke['selected']:
  hits=[r for r in rows if all(str(r[k])==str(wanted[k]) for k in ('task','qid','group_id','message_sha256','image_sha256'))]
  if len(hits)!=1:raise ValueError('smoke anchor input differs')
  anchor_rows.append(hits[0])
 code=[Path(p) for p in local('source_inventory').sources()]
 frozen={'GT_opened':False,'scoring_performed':False,'approval':approval,'e1_proof_sha256':sha(a.e1_acceptance),'upstream':upstream,'plan_sha256':sha(PLAN),'full_plan_sha256':sha(HERE/'plan.json'),'weights_identity_sha256':sha(WEIGHTS),'weight_stats_before':stats,'anchors':[{k:r[k] for k in ('task','qid','group_id','image_sha256','message_sha256')} for r in anchor_rows],'code_sha256':{str(p):sha(p) for p in code}}
 with (out/'frozen.json').open('x') as f:json.dump(frozen,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 wrapper=Wrapper(**plan['recipe'])
 if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('backend mismatch')
 def generate(w,row):
  with torch.inference_mode():return BaseWrapper.generate_inner_transformers_fixed(w,row['message'],dataset=row['task'])
 def invoke(row,family,arm,journal,archive,need_logits):return runner.run_arm(torch,wrapper,pruner,row,family,arm,generate,journal,archive,capture_prefill_logits=need_logits)
 def archive(tensors,path):
  import numpy as np
  arrays={k:v.detach().contiguous().cpu().view(torch.uint8).numpy() for k,v in tensors.items()}
  with path.open('xb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
  return {'kind':'full_arrays','path':str(path),'sha256':sha(path)}
 def summary(tensors):
  from identity_records import tensor_meta
  return {k:tensor_meta(v) for k,v in tensors.items()}
 run_panel(rows=rows,e1_refs=refs,anchors=anchor_rows,out=out/'data',invoke=invoke,archive=archive,summary=summary)
 for path,before in stats.items():
  s=Path(path).stat()
  if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=before:raise ValueError('weights changed')
 if any(sha(path)!=digest for path,digest in frozen['code_sha256'].items()):raise ValueError('full panel source changed')
 with (out/'worker_complete.json').open('x') as f:
  json.dump({'questions':384,'physical_generations':3840,'weight_stats_unchanged':True,'source_unchanged':True},f);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
