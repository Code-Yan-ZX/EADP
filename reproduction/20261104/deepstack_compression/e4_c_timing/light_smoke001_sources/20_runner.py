"""E4-C runner primitives; real path remains fail-closed before GPU access.

 Interleaved latency uses one process/model and changes mode between samples.
 Memory measurements run in a fresh single-mode process and are never pooled
 into latency statistics. Actual C-SKIP and P3 identity approval is mandatory.
"""
import argparse,json,os,sys,time
from pathlib import Path
from timer import measure
from identity_gate import require_gpu_identity

def execute_one(wrapper,pruner_module,torch,row,mode,protocol,synchronize):
 """Synchronous author generation including image decode/preprocess after T0."""
 from light_adapter import LightAdapter
 from transformers.generation.stopping_criteria import StoppingCriteriaList
 if protocol not in ('natural','fixed32'):raise ValueError('unsupported generation protocol')
 original=wrapper.model.generate;captured={}
 def invoke(boundary):
  def generate(*args,**kwargs):
   kwargs['stopping_criteria']=StoppingCriteriaList([boundary])
   if protocol=='fixed32':kwargs['min_new_tokens']=32;kwargs['max_new_tokens']=32
   result=original(*args,**kwargs);captured['result']=result
   return result
  wrapper.model.generate=generate
  try:
   from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatCDPruner
   return Qwen3VLChatCDPruner.generate_inner_transformers_fixed(wrapper,row['message'],dataset=row['task'])
  finally:wrapper.model.generate=original
 # Install intervention outside measured window in every arm, with identity
 # correctness proven separately. No hashes or observers inside timed region.
 with LightAdapter(torch,wrapper,pruner_module,mode,row['image_start']) as adapter:
  answer,times=measure(invoke,synchronize)
  if not adapter.state['prefill_seen']:raise ValueError('prefill was not reached')
 tokens=captured['result'].detach().cpu().tolist()[0]
 if len(tokens)<1 or (protocol=='fixed32' and len(tokens)!=32):raise ValueError('invalid generation length')
 # This author wrapper calls generate(inputs_embeds=...), whose returned IDs
 # are continuations only; verify against B identity before actual use.
 return {'mode':mode,'protocol':protocol,'answer':answer,'output_ids':tokens,'output_tokens':len(tokens),'boundaries':times,'instrumented_logits_ready':False}

def execute_schedule(schedule,rows,role,mode,execute,journal,excluded=()):
 """Replay the frozen schedule with one model; durability outside measured window.

 For memory, the caller must launch one fresh process per mode; process/GPU
 identity and peak sampling must be verified externally, not inferred here.
 """
 from contract import validate_schedule,make_schedule
 validate_schedule(schedule)
 if schedule!=make_schedule(rows,excluded):raise ValueError('schedule differs from preregistered seeded order')
 if role not in ('latency_interleaved','memory_isolated'):raise ValueError('unknown role')
 if (role=='latency_interleaved' and mode is not None) or (role=='memory_isolated' and mode not in ('C-ON','C-DROP','C-SKIP')):raise ValueError('invalid role mode')
 if len(rows)!=384 or any(r.get('split')!='DEV' for r in rows):raise ValueError('full DEV input required')
 bykey={(r['task'],str(r['qid'])):r for r in rows}
 if len(bykey)!=len(rows):raise ValueError('duplicate input')
 jobs=schedule['schedule'] if mode is None else [x for x in schedule['schedule'] if x['mode']==mode]
 for job in jobs:
  row=bykey.get((job['task'],job['qid']))
  if row is None or row.get('group_id')!=job['group_id'] or row.get('split')!='DEV':raise ValueError('schedule/input mismatch')
 # Refuse bad later entries before doing any actual measurements.
 for job in jobs:
  row=bykey[(job['task'],job['qid'])]
  try:value=execute(row,job['mode'],job['protocol'])
  except Exception as exc:
   journal({'status':'failed','job':job,'worker_pid':os.getpid(),'error_type':type(exc).__name__,'error':str(exc)})
   raise
  journal({'status':'generated','job':job,'worker_pid':os.getpid(),'measurement_role':role,'result':value})
  if value.get('mode')!=job['mode'] or value.get('protocol')!=job['protocol']:raise ValueError('executor response mismatch')
  journal({'status':'completed','job':job,'worker_pid':os.getpid(),'measurement_role':role})
 return len(jobs)

def run_model_jobs(schedule,rows,role,mode,wrapper,native,torch,journal,excluded=()):
 """Single loaded model; caller must establish exclusive device/process identity.

 NVML sampling and GPU identity are external prerequisites, not claims made by
 this worker. Memory-only hooks are never used in the latency role.
 """
 from memory_probe import memory_measure
 def one(row,selected,protocol):
  if type(row.get('image_start')) is not int:raise ValueError('frozen image token start required')
  def run():
   with torch.inference_mode():return execute_one(wrapper,native,torch,row,selected,protocol,torch.cuda.synchronize)
  if role=='memory_isolated':
   value,stats=memory_measure(torch,wrapper.model,run);value['memory_only']={**stats,'kv_cache_bytes':stats['kv_cache_bytes_final']}
  else:value=run()
  return value
 return execute_schedule(schedule,rows,role,mode,one,journal,excluded)

def durable_journal(path):
 """A new run only: never silently resume or discard partial measurements."""
 f=Path(path).open('x',encoding='utf-8')
 def append(record):
  f.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
 return f,append

def fake_process(jobs,mode=None):
 """Testing of process-role/schedule invariants only; synthetic inputs."""
 for job in jobs:
  if mode is not None and job['mode']!=mode:raise ValueError('memory worker cannot mix configurations')
  yield {'mode':job['mode'],'order':job['order'],'pid':os.getpid(),'fake':True}

def main():
 p=argparse.ArgumentParser();p.add_argument('--role',choices=('latency_interleaved','memory_isolated'),required=True);p.add_argument('--mode',choices=('C-ON','C-DROP','C-SKIP'));p.add_argument('--fake',action='store_true');a=p.parse_args()
 if a.role=='memory_isolated' and a.mode is None:raise ValueError('memory worker requires one configuration')
 if a.role=='latency_interleaved' and a.mode is not None:raise ValueError('interleaved latency may not pin one configuration')
 if not a.fake:
  # This executes before importing torch, author model, or initializing GPU.
  require_gpu_identity()
  raise AssertionError('unreachable until separate identity review')
 rows=[json.loads(line) for line in sys.stdin if line.strip()]
 for value in fake_process(rows,a.mode):print(json.dumps(value),flush=True)
if __name__=='__main__':main()
