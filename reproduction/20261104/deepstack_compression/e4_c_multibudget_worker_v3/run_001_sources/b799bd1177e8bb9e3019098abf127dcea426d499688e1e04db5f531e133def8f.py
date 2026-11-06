"""Strict no-GPU provenance and GPU-prelaunch proof checks."""
import hashlib,importlib.util,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;V2=BASE/'e4_c_budget_pilot_v2';REVIEW=BASE/'e1_independent_review'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def terminal(start,end):
 s=read(start);t=read(end)
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']!=t.get('worker_pid'):raise ValueError('supervised process not completed')
 return s
def resolved_command(s):
 cwd=Path(s['cwd']);return [x if x.startswith('-') else str((cwd/x).resolve()) if not Path(x).is_absolute() else str(Path(x).resolve()) for x in s['command']]

def prior():
 path=BASE/'e4_c_budget_runtime/gate.py';spec=importlib.util.spec_from_file_location('_budget_prior_gate',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 result=mod.validate_prior()
 # Bind the actual independent reviewer process and output, not just proof text.
 start=REVIEW/'monitor_e4c_budget_pilot_v2_preflight/started.json';end=REVIEW/'monitor_e4c_budget_pilot_v2_preflight/terminal.json'
 s=terminal(start,end);cmd=resolved_command(s);reviewer=REVIEW/'review_e4c_budget_pilot_v2_preflight.py';out=REVIEW/'e4c_budget_pilot_v2_preflight_review.json'
 if cmd.count(str(reviewer))!=1 or cmd.count('--out')!=1 or cmd[cmd.index('--out')+1]!=str(out):raise ValueError('wrong prior reviewer command')

 cpu=read(HERE/'cpu002.json')
 terminal(HERE/'monitor_cpu002/started.json',HERE/'monitor_cpu002/terminal.json')
 if cpu.get('status')!='passed' or cpu.get('GPU_authorized') is not False or cpu.get('model_loaded') is not False:raise ValueError('fixed32 CPU proof missing')
 for path,digest in cpu['source_sha256'].items():
  if sha(path)!=digest:raise ValueError('fixed32 CPU tested source changed '+path)
 identity_path=BASE/'e4_c_budget_ttft_pilot_v3/identity_gate.py'
 spec=importlib.util.spec_from_file_location('_fixed32_old_identity',identity_path);identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
 identity.admit_identity()
 return result

def plan():
 p=read(HERE/'execution_plan005.json')
 if p.get('stage')!='4-budget fixed32 24-generation diagnostic' or p.get('budgets')!=[128,256,512,1024]:raise ValueError('wrong plan')
 for path,digest in p['source_sha256'].items():
  if sha(path)!=digest:raise ValueError('source changed '+path)
 return p

def execution_cpu(p):
 path=HERE/'execution_cpu005.json';started=HERE/'monitor_execution_cpu005/started.json';ended=HERE/'monitor_execution_cpu005/terminal.json'
 proof=read(path);s=terminal(started,ended);cmd=resolved_command(s)
 if cmd.count(str(HERE/'freeze_execution.py'))!=1 or (proof.get('status'),proof.get('test_returncode'),proof.get('model_loaded'),proof.get('GPU_authorized'),proof.get('predictions_opened'))!=('passed',0,False,False,True) or proof.get('plan_sha256')!=sha(HERE/'execution_plan005.json') or proof.get('source_sha256')!=p['source_sha256']:raise ValueError('fixed32 execution CPU proof mismatch')
 return proof

def admit_cpu_preflight(p):
 execution_cpu(p)
 proof=REVIEW/'e4c_multibudget_fixed32_worker_preflight_review.json';reviewer=REVIEW/'review_e4c_multibudget_fixed32_worker_preflight.py'
 if not proof.is_file():raise ValueError('CPU container locked: independent preflight review missing')
 r=read(proof);start=REVIEW/'monitor_e4c_multibudget_fixed32_worker_preflight/started.json';end=REVIEW/'monitor_e4c_multibudget_fixed32_worker_preflight/terminal.json';s=terminal(start,end)
 cmd=resolved_command(s)
 fields={'plan_sha256':HERE/'execution_plan005.json','execution_cpu_sha256':HERE/'execution_cpu005.json','execution_cpu_started_sha256':HERE/'monitor_execution_cpu005/started.json','execution_cpu_terminal_sha256':HERE/'monitor_execution_cpu005/terminal.json','reviewer_sha256':reviewer}
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('scope')!='fixed32_cpu_preflight_only' or r.get('errors')!=[] or r.get('GT_opened') is not False or r.get('predictions_opened') is not True or r.get('scoring_performed') is not False or r.get('source_sha256')!=p['source_sha256'] or any(r.get(k)!=sha(v) for k,v in fields.items()) or cmd.count(str(reviewer))!=1 or cmd.count('--out')!=1 or cmd[cmd.index('--out')+1]!=str(proof):raise ValueError('CPU preflight review mismatch')
 return r

def admit_gpu():
 """Require future independent review and real retained no-GPU worker preflight."""
 p=plan();old=prior();admit_cpu_preflight(p);proof=REVIEW/'e4c_multibudget_fixed32_prelaunch_review.json';reviewer=REVIEW/'review_e4c_multibudget_fixed32_prelaunch.py'
 if not proof.is_file():raise ValueError('GPU locked: independent prelaunch proof missing')
 r=read(proof);rp=HERE/'run_preflight001';ms=HERE/'monitor_preflight001/started.json';mt=HERE/'monitor_preflight001/terminal.json'
 fields={'plan_sha256':HERE/'execution_plan005.json','prior_review_sha256':REVIEW/'e4c_budget_pilot_v2_preflight_review.json',
         'worker_preflight_sha256':rp/'worker_preflight.json','processor_preflight_sha256':rp/'processor_preflight.json',
         'preflight_launch_sha256':rp/'launch.json','preflight_exit_sha256':rp/'exit.json','preflight_started_sha256':ms,
         'preflight_terminal_sha256':mt,'reviewer_sha256':reviewer,'execution_cpu_sha256':HERE/'execution_cpu005.json',
         'execution_cpu_started_sha256':HERE/'monitor_execution_cpu005/started.json','execution_cpu_terminal_sha256':HERE/'monitor_execution_cpu005/terminal.json'}
 if (r.get('status')!='passed' or r.get('independent_review') is not True or r.get('scope')!='gpu_fixed32_diagnostic_prelaunch' or r.get('errors')!=[] or
     r.get('GT_opened') is not False or r.get('scoring_performed') is not False or r.get('source_sha256')!=p['source_sha256'] or
     any(r.get(k)!=sha(v) for k,v in fields.items())):raise ValueError('GPU prelaunch proof mismatch')
 terminal(ms,mt);s=terminal(REVIEW/'monitor_e4c_multibudget_fixed32_prelaunch/started.json',REVIEW/'monitor_e4c_multibudget_fixed32_prelaunch/terminal.json')
 cmd=resolved_command(s)
 if cmd.count(str(reviewer))!=1 or cmd.count('--out')!=1 or cmd[cmd.index('--out')+1]!=str(proof):raise ValueError('wrong prelaunch reviewer command')
 w=read(rp/'worker_preflight.json');ex=read(rp/'exit.json');state=ex.get('last_inspect') or {}
 if (w.get('status')!='passed' or w.get('cuda_initialized') is not False or w.get('model_loaded') is not False or
     w.get('plan_sha256')!=sha(HERE/'execution_plan005.json') or w.get('prior_review_sha256')!=old['prior_proof_sha256'] or
     ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('timed_out') is not False or ex.get('interrupted') is not None or
     ex.get('client_exited_while_container_running') is not False or ex.get('container_stopped_verified') is not True or
     state.get('running') is not False or state.get('pid')!=0 or state.get('host_devices') or state.get('device_requests')):raise ValueError('invalid worker preflight')
 return {'plan_sha256':sha(HERE/'execution_plan005.json'),'gpu_review_sha256':sha(proof),'source_sha256':p['source_sha256']}

def check_supervisor(s,runid,now,plan):
 out=HERE/('run_'+runid);progress=out/'diagnostic/journal.jsonl';cmd=resolved_command(s)
 if (s.get('worker_pid')!=os.getpid() or s.get('process_group')!=os.getpgrp() or os.getpid()!=os.getpgrp() or
     s.get('timeout_seconds')!=plan['outer_timeout_seconds'] or s.get('stale_seconds')!=plan['stale_seconds'] or
     s.get('progress_paths')!=[str(progress)] or s.get('snapshot_seconds')!=3600 or s.get('poll_seconds')!=5 or
     s.get('grace_seconds')!=plan['grace_seconds'] or cmd.count(str(HERE/'launch.py'))!=1 or
     s['command'].count('--run-id')!=1 or s['command'][s['command'].index('--run-id')+1]!=runid):raise ValueError('GPU supervisor mismatch')
 initial=read(BASE/'initial_plan.json')
 if sha(BASE/'initial_plan.json')!=plan['budget_source_sha256'] or now+plan['outer_timeout_seconds']+plan['acceptance_reserve_seconds']>initial['budget_end_unix'] or now<initial['budget_start_unix']:raise ValueError('user deadline insufficient')
 return True
