"""Fail-closed admission to the 16-arm DEV diagnostic; no GPU operations."""
import json
from pathlib import Path
from real_worker import HERE,BASE,PLAN,WEIGHTS,sha

REVIEW=BASE/'e1_independent_review/e4c_light_prelaunch_review.json'
REVIEWER=BASE/'e1_independent_review/review_e4c_light_prelaunch.py'
REVIEW_STARTED=BASE/'e1_independent_review/monitor_e4c_light_prelaunch/started.json'
REVIEW_TERMINAL=BASE/'e1_independent_review/monitor_e4c_light_prelaunch/terminal.json'
PRE_REVIEW=BASE/'e1_independent_review/e4c_light_worker_preflight_review.json'
PRE_REVIEWER=BASE/'e1_independent_review/review_e4c_light_worker_preflight.py'
PRE_STARTED=BASE/'e1_independent_review/monitor_e4c_light_worker_preflight/started.json'
PRE_TERMINAL=BASE/'e1_independent_review/monitor_e4c_light_worker_preflight/terminal.json'

def sources():
 local=('light_admission.py','light_smoke_worker.py','light_smoke_launch.py','light_smoke_budget.json','light_smoke_core.py','light_smoke_plan.py',
        'light_arm.py','light_capture.py','light_adapter.py','light_compare.py','light_verify_records.py','light_verify_core.py','light_verify.py',
        'light_reference_gate.py','light_container_probe.py','light_preflight_launch.py','identity_gate.py','real_worker.py','input_preflight.py','timing_launch.py',
        'runner.py','timer.py','contract.py','e1_reference_gate.py')
 paths=[HERE/x for x in local]+[BASE/'e4'/x for x in ('budget_runner.py','budget_adapter.py','off_adapter.py','identity_records.py','decode_observer.py','skip_deepstack.py')]
 runtime=Path(json.loads(PLAN.read_text())['container_runtime_source'])
 return paths+[BASE/'monitor/supervise.py',PLAN,WEIGHTS,PLAN.parent/'run_panel.py',PLAN.parent/'diagnose.py',runtime]

def terminal(started,terminal):
 s=json.loads(Path(started).read_text());t=json.loads(Path(terminal).read_text())
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid'):
  raise ValueError('supervised process incomplete')
 return s

def reviewer_command(started,reviewer=REVIEWER,review=REVIEW):
 command=started.get('command',[]);cwd=Path(started.get('cwd','/'))
 if ([(cwd/x).resolve() for x in command if Path(x).name==reviewer.name]!=[reviewer.resolve()] or
     command.count('--out')!=1 or command.index('--out')+1>=len(command) or (cwd/command[command.index('--out')+1]).resolve()!=review.resolve()):
  raise ValueError('wrong reviewer process or output')

def require_source_coverage(observed,expected,label):
 if not isinstance(observed,dict) or any(observed.get(path)!=digest for path,digest in expected.items()):
  raise ValueError(label+' source inventory incomplete or changed')

def admit(review_path=None,preflight_only=False):
 """The reviewer must audit actual source, CPU proof and retained preflight."""
 review,reviewer,review_started,review_terminal=(PRE_REVIEW,PRE_REVIEWER,PRE_STARTED,PRE_TERMINAL) if preflight_only else (REVIEW,REVIEWER,REVIEW_STARTED,REVIEW_TERMINAL)
 if review_path is None:review_path=review
 review_path=Path(review_path).resolve()
 if review_path!=review.resolve():raise ValueError('unapproved reviewer proof path')
 if not review_path.is_file():raise ValueError('identity smoke locked: independent review missing')
 r=json.loads(review_path.read_text())
 if (r.get('status')!='passed' or r.get('independent_review') is not True or r.get('GT_opened') is not False or r.get('scoring_performed') is not False or r.get('errors')!=[] or
     r.get('scope')!=('worker_preflight_only' if preflight_only else 'gpu_diagnostic_prelaunch')):
  raise ValueError('prelaunch independent review failed')
 expected={str(p):sha(p) for p in sources()}
 if r.get('source_sha256')!=expected or r.get('reviewer_sha256')!=sha(reviewer):raise ValueError('review source identity mismatch')
 names=['reference','reference_started','reference_terminal','cpu_proof','cpu_started','cpu_terminal','preflight_launch','preflight_exit','preflight_started','preflight_terminal']
 if not preflight_only:names+=['worker_preflight_launch','worker_preflight_exit','worker_preflight_started','worker_preflight_terminal','worker_preflight_result']
 for name in names:
  path=Path(r[name+'_path']).resolve()
  if r.get(name+'_sha256')!=sha(path):raise ValueError('review input changed: '+name)
  if not path.is_relative_to(HERE):raise ValueError('review input outside E4-C')
 terminal(r['reference_started_path'],r['reference_terminal_path']);terminal(r['cpu_started_path'],r['cpu_terminal_path']);terminal(r['preflight_started_path'],r['preflight_terminal_path'])
 cpu=json.loads(Path(r['cpu_proof_path']).read_text())
 if cpu.get('status')!='passed' or cpu.get('returncode')!=0 or cpu.get('source_sha256_before')!=cpu.get('source_sha256_after'):raise ValueError('CPU proof invalid')
 require_source_coverage(cpu['source_sha256_after'],expected,'CPU proof')
 pre=json.loads(Path(r['preflight_launch_path']).read_text());ex=json.loads(Path(r['preflight_exit_path']).read_text());state=(ex.get('inspect') or {}).get('state') or {}
 if (pre.get('gpu_requested') is not False or '--device' in pre.get('command',[]) or '--gpus' in pre.get('command',[]) or
     ex.get('returncode')!=0 or ex.get('error') is not None or ex.get('interrupted') is not None or ex.get('timed_out') is not False or ex.get('source_unchanged') is not True or
     ex.get('client_exited_while_container_running') is not False or ex.get('container_stopped_verified') is not True or ex.get('cleanup_error') is not None or
     state.get('Running') is not False or state.get('Pid')!=0 or state.get('ExitCode')!=0 or not ex.get('identity') or
     ex['identity'].get('container_id')!=(ex.get('inspect') or {}).get('id') or (ex.get('inspect') or {}).get('host_devices') or (ex.get('inspect') or {}).get('device_requests')):raise ValueError('CPU container preflight invalid')
 require_source_coverage(pre.get('source_sha256'),expected,'CPU preflight')
 if not preflight_only:
  wstarted=terminal(r['worker_preflight_started_path'],r['worker_preflight_terminal_path'])
  wp=json.loads(Path(r['worker_preflight_launch_path']).read_text());we=json.loads(Path(r['worker_preflight_exit_path']).read_text());wr=json.loads(Path(r['worker_preflight_result_path']).read_text());ws=(we.get('inspect') or {}).get('state') or {}
  cmd=wp.get('command',[]);workdir=Path(r['worker_preflight_launch_path']).resolve().parent;plan=json.loads(PLAN.read_text());pre_approval=admit(preflight_only=True)
  if (wp.get('worker_preflight') is not True or wp.get('approval_bundle')!=pre_approval or wp.get('worker_sha256')!=expected[str(HERE/'light_smoke_worker.py')] or
      (wp.get('controller') or {}).get('pid')!=wstarted.get('worker_pid') or Path(r['worker_preflight_exit_path']).resolve().parent!=workdir or
      Path(r['worker_preflight_result_path']).resolve()!=workdir/'worker_preflight.json' or cmd.count('--preflight-only')!=1 or
      cmd.count(str(HERE/'light_smoke_worker.py'))!=1 or cmd.count('--references')!=1 or cmd.index('--references')+1>=len(cmd) or
      Path(cmd[cmd.index('--references')+1]).resolve()!=Path(r['reference_path']).resolve() or cmd.count('--out')!=1 or cmd.index('--out')+1>=len(cmd) or Path(cmd[cmd.index('--out')+1]).resolve()!=workdir or
      wp.get('gpu_requested') is not False or '--device' in cmd or '--gpus' in cmd or
      we.get('returncode')!=0 or we.get('error') is not None or we.get('timed_out') is not False or we.get('interrupted') is not None or we.get('source_unchanged') is not True or
      we.get('client_exited_while_container_running') is not False or we.get('container_stopped_verified') is not True or we.get('cleanup_error') is not None or
      ws.get('Running') is not False or ws.get('Pid')!=0 or ws.get('ExitCode')!=0 or not we.get('identity') or we['identity'].get('container_id')!=(we.get('inspect') or {}).get('id') or
      (we.get('inspect') or {}).get('image')!=plan['container_image'] or (we.get('inspect') or {}).get('host_devices') or (we.get('inspect') or {}).get('device_requests') or wr.get('status')!='passed' or
      wr.get('cuda_initialized') is not False or wr.get('model_loaded') is not False or wr.get('predictions') is not False or wr.get('reference_sha256')!=r['reference_sha256'] or
      wr.get('source_sha256')!=expected or wr.get('plan_sha256')!=sha(PLAN) or wr.get('weights_identity_sha256')!=sha(WEIGHTS) or wr.get('versions')!=plan['expected_versions'] or
      len(wr.get('weight_stats_before',{}))!=4):raise ValueError('full CPU worker preflight invalid')
  require_source_coverage(wp.get('source_sha256'),expected,'full CPU worker preflight')
 ref=json.loads(Path(r['reference_path']).read_text())
 if ref.get('status')!='passed' or len(ref.get('rows',[]))!=3 or len(ref.get('plan',{}).get('jobs',[]))!=16:raise ValueError('DEV reference invalid')
 started=terminal(review_started,review_terminal)
 reviewer_command(started,reviewer,review)
 return {'review_sha256':sha(review_path),'review_started_sha256':sha(review_started),'review_terminal_sha256':sha(review_terminal),'preflight_only':preflight_only,
         'reference_path':r['reference_path'],'reference_sha256':r['reference_sha256'],'source_sha256':expected}
