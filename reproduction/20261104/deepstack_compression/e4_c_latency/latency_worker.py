"""Prepared exclusive-GPU batch-1 latency worker; admission remains locked.

The 16-arm identity proof, reviewed bounded launcher, completed latency smoke
and independent acceptance are prerequisites for the full stage. None of the
entry points here can initialize a GPU until those gates are implemented.
"""
import argparse,importlib.metadata,inspect,json,os,sys,time
from pathlib import Path
from latency_gate import require_light_identity
from real_worker import HERE,BASE,PLAN,WEIGHTS,sha,source_and_weights,preflight_one,references_to_rows
from input_preflight import load_accepted_e1_references
from schedule import build
from runner import durable_journal
from stage_runner import execute_model

def sources():
 names=('latency_worker.py','latency_gate.py','latency_admission.py','latency_launch.py','latency_verify.py','preflight_worker.py','preflight_import.py','import_probe.py','cpu_check.py','test_latency.py','test_lifecycle.py',
        'schedule.py','stage_runner.py','runner.py','timer.py','contract.py','input_preflight.py','e1_reference_gate.py',
        'identity_gate.py','light_adapter.py','real_worker.py','memory_probe.py','timing_supervise.py','interference.py','timing_launch.py','verify_core.py','budget_plan.json')
 runtime=Path(json.loads(PLAN.read_text())['container_runtime_source'])
 return [HERE/x for x in names]+[BASE/'e4/skip_deepstack.py',BASE/'monitor/supervise.py',PLAN,WEIGHTS,BASE/'initial_plan.json',PLAN.parent/'run_panel.py',PLAN.parent/'diagnose.py',runtime]

def prepare(stage):
 e1=BASE/'e1';proof=e1/'e1full001_completion.json';started=e1/'monitorverifye1full001/started.json';terminal=e1/'monitorverifye1full001/terminal.json'
 evidence={str(x):sha(x) for x in (proof,started,terminal)}
 rows,refs=load_accepted_e1_references(proof,started,terminal);rows=references_to_rows(rows,refs)
 anchor=BASE/'data_manifest/anchor_exclusion/anchor_groups.jsonl';anchor_before=sha(anchor);excluded={json.loads(x)['group_id'] for x in anchor.read_text().splitlines() if x.strip()}
 if len(excluded)!=16:raise ValueError('old anchor exclusion identity')
 schedule=build(rows,excluded,stage);plan=json.loads(PLAN.read_text());recipe=plan['recipe']
 expected={'visual_token_num':256,'alpha':0.5,'beta':2.0,'min_pixels':1048576,'max_pixels':1048576,'max_new_tokens':2048,'use_custom_prompt':True,'use_vllm':False}
 if any(recipe.get(k)!=v for k,v in expected.items()):raise ValueError('author recipe changed')
 frozen={str(p):sha(p) for p in sources()};weights=source_and_weights(plan)
 if sha(anchor)!=anchor_before or any(sha(path)!=digest for path,digest in evidence.items()) or any(sha(path)!=digest for path,digest in frozen.items()):raise ValueError('source or E1 proof changed during preflight')
 return {'rows':rows,'refs':refs,'excluded':excluded,'schedule':schedule,'plan':plan,'weights_before':weights,'source_sha256':frozen,
         'e1_acceptance_sha256':sha(proof),'e1_started_sha256':sha(started),'e1_terminal_sha256':sha(terminal),'anchor_sha256':sha(anchor)}

def load_runtime(prepared):
 plan=prepared['plan'];sys.path[:0]=[plan['author_root'],str(Path(plan['author_root'])/'VLMEvalKit')]
 import torch
 from model import pruner
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP,Qwen3VLChatCDPruner
 versions={n:importlib.metadata.version(n) for n in plan['expected_versions']}
 expected_pruner=(Path(plan['author_root'])/'model/pruner.py').resolve();expected_wrapper=(Path(plan['author_root'])/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py').resolve()
 if (versions!=plan['expected_versions'] or Path(pruner.__file__).resolve()!=expected_pruner or Path(inspect.getfile(Qwen3VLChatEADP)).resolve()!=expected_wrapper or
     Path(inspect.getfile(Qwen3VLChatCDPruner)).resolve()!=expected_wrapper):raise ValueError('author runtime import mismatch')
 return torch,pruner,Qwen3VLChatEADP,versions

def save(path,value):
 with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def continue_stage(args,prepared,identity):
 """Continuation for later independently reviewed admission, not a CLI bypass."""
 out=args.out.resolve()
 if not out.is_dir() or any((out/x).exists() for x in ('frozen.json','journal.jsonl','worker_complete.json')):raise ValueError('invalid/reused timing output')
 torch,native,wrapper_class,versions=load_runtime(prepared)
 if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
 wrapper=wrapper_class(**prepared['plan']['recipe'])
 if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('backend mismatch')
 selected={(x['task'],str(x['qid'])) for x in prepared['schedule']['schedule']}
 for row in prepared['rows']:
  key=(row['task'],str(row['qid']))
  if key in selected:preflight_one(torch,wrapper,row,prepared['refs'][key])
 import gc
 gc.collect();torch.cuda.synchronize()
 if any(sha(path)!=digest for path,digest in prepared['source_sha256'].items()):raise ValueError('source changed during model load')
 if not out.is_dir() or any((out/x).exists() for x in ('frozen.json','journal.jsonl','worker_complete.json')):raise ValueError('invalid/reused timing output')
 frozen={k:prepared[k] for k in ('schedule','source_sha256','e1_acceptance_sha256','e1_started_sha256','e1_terminal_sha256','anchor_sha256')}
 frozen.update(stage=args.stage,identity=identity,weight_stats_before=prepared['weights_before'],versions=versions,
               scope='DEV batch1 warm-cache latency; no GT or scores',GT_opened=False,scoring_performed=False)
 save(out/'frozen.json',frozen);handle,journal=durable_journal(out/'journal.jsonl')
 try:count=execute_model(prepared['schedule'],prepared['rows'],prepared['excluded'],args.stage,wrapper,native,torch,journal)
 finally:handle.close()
 if count!=len(prepared['schedule']['schedule']):raise ValueError('timing schedule incomplete')
 for path,before in prepared['weights_before'].items():
  s=Path(path).stat()
  if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=before:raise ValueError('weight identity changed')
 if any(sha(path)!=digest for path,digest in prepared['source_sha256'].items()):raise ValueError('timing source changed')
 save(out/'worker_complete.json',{'status':'completed','stage':args.stage,'jobs':count,'finished_unix':time.time()})

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=('smoke','full'),required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');args=p.parse_args()
 from latency_admission import admit
 approval=admit(args.stage,preflight_only=args.preflight_only) # blocks before model/GPU
 out=args.out.resolve();launch=json.loads((out/'launch.json').read_text())
 if launch.get('approval_bundle')!=approval or launch.get('worker_sha256')!=sha(__file__):raise ValueError('latency launcher identity mismatch')
 prepared=prepare(args.stage)
 if prepared['source_sha256']!=approval['source_sha256']:raise ValueError('latency reviewed source changed')
 if args.preflight_only:
  torch,_native,_wrapper,versions=load_runtime(prepared)
  if torch.cuda.is_initialized():raise ValueError('CPU preflight initialized CUDA')
  result={'status':'passed','stage':args.stage,'cuda_initialized':False,'model_loaded':False,'predictions':False,'light_identity':approval['light_identity'],
          'source_sha256':prepared['source_sha256'],'weight_stats_before':prepared['weights_before'],'versions':versions,
          'e1_acceptance_sha256':prepared['e1_acceptance_sha256'],'anchor_sha256':prepared['anchor_sha256'],
          'schedule':prepared['schedule'],'scope':'complete latency worker before model load; no GPU, predictions, GT or scoring'}
  save(out/'worker_preflight.json',result);return
 continue_stage(args,prepared,approval['light_identity'])
if __name__=='__main__':main()
