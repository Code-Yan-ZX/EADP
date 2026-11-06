"""Injected six-arm controller prototype. No GPU/container entry is provided.

A future reviewed bounded runner must supply authenticated container identity,
real exit state and separately recorded source/weight evidence. Tests here use
only a fake runner and do not constitute model or GPU admission.
"""
import json,os
from pathlib import Path
import schedule_contract as contract

def emit(path,value):
 with Path(path).open('a') as f:
  f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())

def execute_with_injected_runner(images,journal,runner):
 jobs=contract.schedule(images);journal=Path(journal)
 if journal.exists():raise ValueError('existing identity journal')
 history=[]
 for job in jobs:
  # This event precedes the runner call so a killed process exposes the arm.
  emit(journal,{'event':'arm_invoked','job':job})
  try:
   result=runner(job)
   if not isinstance(result,dict):raise ValueError('runner result not structured')
   cid=result.get('container_id');started={'event':'started','job':job,'container_id':cid}
   complete={'event':'completed','job':job,'container_id':cid,'returncode':result.get('returncode'),
    'container_exit_code':result.get('container_exit_code'),'container_stopped_verified':result.get('container_stopped_verified'),
    'timed_out':result.get('timed_out'),'interrupted':result.get('interrupted'),'result_status':result.get('result_status')}
   history.extend((started,complete));emit(journal,{'event':'arm_result','job':job,'result':result})
   # Check each arm immediately, and never silently continue past failed work.
   # Container uniqueness is checked across the full actual history below.
   if (not isinstance(cid,str) or len(cid)!=64 or any(c not in '0123456789abcdef' for c in cid)
       or cid in {x['container_id'] for x in history[:-2:2]}
       or complete['returncode']!=0 or complete['container_exit_code']!=0
       or complete['container_stopped_verified'] is not True or complete['timed_out'] is not False
       or complete['interrupted'] is not None or complete['result_status']!='passed'):
    raise ValueError('identity arm failed or container reused')
  except BaseException as exc:
   emit(journal,{'event':'arm_failed','job':job,'error_type':type(exc).__name__,'error':str(exc)})
   raise
 contract.validate_history(jobs,history)
 emit(journal,{'event':'all_arms_completed','images':3,'jobs':6})
 return history

def main():
 import argparse,importlib.util,shutil,sys,time
 D=Path(__file__).resolve().parent;B=D.parents[1]
 ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);a=ap.parse_args()
 if a.run_id!='001':raise ValueError('unplanned run')
 # The gate is deliberately unconditional rejection until independently reviewed.
 cached=sys.modules.get('gpu_admission')
 if cached is not None and (not getattr(cached,'__file__',None) or Path(cached.__file__).resolve()!=D/'gpu_admission.py'):raise ValueError('foreign cached GPU gate')
 import gpu_admission
 if Path(gpu_admission.__file__).resolve()!=D/'gpu_admission.py':raise ValueError('foreign GPU gate')
 approval=gpu_admission.require_gpu()
 plan=json.loads((D/'execution_plan001.json').read_text());base=json.loads((B/'e6/pixel_wavefront_model_t020/preflight_plan002.json').read_text());jobs=contract.schedule(base['images'])
 if plan.get('jobs')!=jobs or plan.get('threshold')!=.2:raise ValueError('unplanned identity schedule')
 monitor=D/'monitor_run001';s=json.loads((monitor/'started.json').read_text());cmd=s['command'];cwd=Path(s['cwd'])
 if (s.get('worker_pid')!=os.getpid() or s.get('process_group')!=os.getpgrp() or os.getpid()!=os.getpgrp()
     or (cwd/cmd[1]).resolve()!=Path(__file__).resolve() or cmd[-2:]!=['--run-id','001']
     or s.get('timeout_seconds')!=plan['outer_timeout_seconds'] or s.get('grace_seconds')!=plan['grace_seconds']
     or s.get('stale_seconds')!=plan['stale_seconds'] or s.get('progress_paths')!=[str(D/'run_001/journal.jsonl')]):raise ValueError('supervisor identity/budget mismatch')
 sys.path.insert(0,str(B/'e4_c_latency'))
 cached=sys.modules.get('timing_launch')
 if cached is not None and (not getattr(cached,'__file__',None) or Path(cached.__file__).resolve()!=B/'e4_c_latency/timing_launch.py'):raise ValueError('foreign cached timing launcher')
 import timing_launch
 if Path(timing_launch.__file__).resolve()!=B/'e4_c_latency/timing_launch.py':raise ValueError('foreign timing launcher')
 initial=json.loads((B/'initial_plan.json').read_text());budget={'budget_source':str(B/'initial_plan.json'),'budget_source_sha256':plan['budget_source_sha256'],'budget_start_unix':initial['budget_start_unix'],'budget_end_unix':initial['budget_end_unix']}
 timing_launch.validate_budget_window(budget,time.time(),plan['inner_timeout_seconds'],plan['outer_timeout_seconds'],plan['grace_seconds'])
 cached=sys.modules.get('bounded_runner_draft')
 if cached is not None and (not getattr(cached,'__file__',None) or Path(cached.__file__).resolve()!=D/'bounded_runner_draft.py'):raise ValueError('foreign cached runner')
 import bounded_runner_draft as runner
 if Path(runner.__file__).resolve()!=D/'bounded_runner_draft.py':raise ValueError('foreign runner')
 if shutil.disk_usage(D).free<plan['minimum_free_bytes']:raise ValueError('insufficient disk space')
 out=D/'run_001';out.mkdir(exist_ok=False)
 cached=sys.modules.get('snapshot_sources')
 if cached is not None and (not getattr(cached,'__file__',None) or Path(cached.__file__).resolve()!=D/'snapshot_sources.py'):raise ValueError('foreign cached snapshot')
 import snapshot_sources
 if Path(snapshot_sources.__file__).resolve()!=D/'snapshot_sources.py':raise ValueError('foreign snapshot module')
 snapshot_sources.snapshot(plan['source_sha256'],out/'sources')
 with (out/'approval.json').open('x') as f:json.dump(approval,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 cached=sys.modules.get('weight_guard')
 if cached is not None and (not getattr(cached,'__file__',None) or Path(cached.__file__).resolve()!=D/'weight_guard.py'):raise ValueError('foreign cached weight guard')
 import weight_guard
 weights=json.loads((B.parents[2]/'reproduction/20261028/qwen/qwen_weight_identity.json').read_text())['files'];before=weight_guard.snapshot(weights)
 with (out/'weights_before.json').open('x') as f:json.dump(before,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 try:
  history=execute_with_injected_runner(base['images'],out/'journal.jsonl',lambda job:runner.run_arm(job,out/f"arm_{job['order']:03d}"))
 finally:
  after=weight_guard.snapshot(weights)
  with (out/'weights_after.json').open('x') as f:json.dump(after,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 if before!=after:raise ValueError('weights changed during run')
 with (out/'complete.json').open('x') as f:json.dump({'status':'passed','jobs':len(history)//2,'weights_stat_unchanged':True,'GT_opened':False,'scoring_performed':False},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
