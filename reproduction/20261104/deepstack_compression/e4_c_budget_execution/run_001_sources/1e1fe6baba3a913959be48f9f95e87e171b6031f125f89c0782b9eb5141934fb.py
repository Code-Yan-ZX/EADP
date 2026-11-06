"""Prepared batch-1 warm-cache timing worker; not yet authorized to execute.

 Entry is unconditionally locked by identity_gate. A future version must bind
 the independent B and lightweight endpoint proofs, approved Docker launcher,
 exclusive-device evidence and (for memory) host NVML sampler handshake.
"""
import argparse,hashlib,importlib.metadata,json,os,sys,time
from pathlib import Path
from identity_gate import require_gpu_identity
from input_preflight import load_accepted_e1_references,validate_processor
from contract import make_schedule
from runner import durable_journal,run_model_jobs

HERE=Path(__file__).resolve().parent;BASE=HERE.parent
ROOT=Path('/media/disk2/YZX/research/EADP_v2')
PLAN=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
WEIGHTS=ROOT/'reproduction/20261028/qwen/qwen_weight_identity.json'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
 return h.hexdigest()

def source_and_weights(plan):
 for group in ('source_sha256','implementation_sha256'):
  for path,expected in plan[group].items():
   if sha(path)!=expected:raise ValueError('author source identity changed')
 items=json.loads(WEIGHTS.read_text())['files']
 if len(items)!=4:raise ValueError('expected four weight shards')
 stats={}
 for item in items:
  p=Path(item['path']);st=p.stat()
  if st.st_size!=item['bytes'] or sha(p)!=item['expected_lfs_sha256']:raise ValueError('weight identity mismatch')
  stats[str(p)]=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
 return stats

def references_to_rows(rows,refs):
 out=[]
 for original in rows:
  r=dict(original);ref=refs[(r['task'],str(r['qid']))]
  if any(r.get(k)!=ref.get(k) for k in ('task','group_id','message_sha256','image_sha256')):raise ValueError('reference/input mismatch')
  r['image_start']=ref['image_start'];out.append(r)
 return out

def preflight_one(torch,wrapper,row,ref):
 """Use the exact author message transformation before processor validation."""
 messages=wrapper._build_messages(row['message'],dataset=row['task'])
 actual=wrapper._processor_inputs(messages)
 try:return validate_processor(torch,actual,ref['processor'],wrapper.model.config.image_token_id,ref['image_start'])
 finally:del actual,messages

def continue_latency(args,rows,refs,old,schedule,plan,weights_before,frozen):
 """Prepared model continuation; not called until external admission exists."""
 if any(sha(path)!=digest for path,digest in frozen.items()):raise ValueError('timing source changed before load')
 sys.path[:0]=[plan['author_root'],str(Path(plan['author_root'])/'VLMEvalKit')]
 import torch
 from model import pruner as native
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatEADP
 if {n:importlib.metadata.version(n) for n in plan['expected_versions']}!=plan['expected_versions']:raise ValueError('runtime versions changed')
 if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
 wrapper=Qwen3VLChatEADP(**plan['recipe'])
 if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('backend changed')
 selected={(x['task'],x['qid']) for x in schedule['schedule']}
 for row in rows:
  key=(row['task'],str(row['qid']))
  if key in selected:
   preflight_one(torch,wrapper,row,refs[key])
 # Release preflight temporaries once; never synchronize each latency job.
 import gc
 gc.collect();torch.cuda.synchronize()
 if any(sha(path)!=digest for path,digest in frozen.items()):raise ValueError('timing source changed during load')
 args.out.mkdir(parents=True,exist_ok=False)
 def save(name,value):
  with (args.out/name).open('x') as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 save('frozen.json',{'source_sha256':frozen,'e1_acceptance_sha256':sha(args.e1_acceptance),'e1_started_sha256':sha(args.e1_started),
  'e1_terminal_sha256':sha(args.e1_terminal),'schedule':schedule,'excluded_old_anchor_groups':sorted(old),'weight_stats_before':weights_before,
  'scope':'DEV batch 1, warm file cache, no GT or score','measurement_role':'latency_interleaved'})
 handle,journal=durable_journal(args.out/'journal.jsonl')
 try:count=run_model_jobs(schedule,rows,'latency_interleaved',None,wrapper,native,torch,journal,old)
 finally:handle.close()
 if count!=len(schedule['schedule']):raise ValueError('incomplete schedule')
 for path,before in weights_before.items():
  st=Path(path).stat()
  if [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]!=before:raise ValueError('weight identity changed')
 if any(sha(path)!=digest for path,digest in frozen.items()):raise ValueError('timing source changed')
 save('worker_complete.json',{'jobs':count,'finished_unix':time.time()})
 return 0

def run_latency(args):
 # Scope remains warm-cache: preflight verifies actual processor tensors once,
 # including image read/decode, before any measured calls.
 rows,refs=load_accepted_e1_references(args.e1_acceptance,args.e1_started,args.e1_terminal)
 rows=references_to_rows(rows,refs)
 anchor=BASE/'data_manifest/anchor_exclusion/anchor_groups.jsonl'
 old={json.loads(x)['group_id'] for x in anchor.read_text().splitlines() if x.strip()}
 if len(old)!=16:raise ValueError('anchor exclusion inventory changed')
 schedule=make_schedule(rows,old);plan=json.loads(PLAN.read_text());recipe=plan['recipe']
 expected={'visual_token_num':256,'alpha':0.5,'beta':2.0,'min_pixels':1048576,'max_pixels':1048576,'max_new_tokens':2048,'use_custom_prompt':True,'use_vllm':False}
 if any(recipe.get(k)!=v for k,v in expected.items()):raise ValueError('author recipe changed')
 weights_before=source_and_weights(plan)
 source=[HERE/x for x in ('real_worker.py','runner.py','timer.py','contract.py','input_preflight.py','e1_reference_gate.py','light_adapter.py','identity_gate.py','timing_supervise.py','interference.py')]
 source+=[BASE/'e4/skip_deepstack.py',BASE/'monitor/supervise.py',PLAN,WEIGHTS]
 frozen={str(p):sha(p) for p in source}
 # Explicit independent external supervisor/launcher and device-exclusivity
 # admission must be implemented before this worker can actually be enabled.
 raise ValueError('real timing locked: reviewed launcher, device and endpoint identity admission required')

 # After admission is independently implemented and reviewed, the explicit
 # continuation is: continue_latency(args,rows,refs,old,schedule,plan,weights_before,frozen).

def main():
 p=argparse.ArgumentParser();p.add_argument('--role',choices=('latency_interleaved','memory_isolated'),required=True);p.add_argument('--mode',choices=('C-ON','C-DROP','C-SKIP'))
 for name in ('e1-acceptance','e1-started','e1-terminal','out'):p.add_argument('--'+name,type=Path,required=True)
 args=p.parse_args()
 if (args.role=='latency_interleaved' and args.mode is not None) or (args.role=='memory_isolated' and args.mode is None):raise ValueError('role/mode mismatch')
 require_gpu_identity() # unconditional, before GPU imports and before predictions
 if args.role=='memory_isolated':raise ValueError('real memory locked: sampler and process handshake required')
 return run_latency(args)
if __name__=='__main__':main()
