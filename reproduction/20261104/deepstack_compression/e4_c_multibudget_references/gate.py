"""Fail-closed planned 248-arm reference provenance; no GPU authority alone."""
import hashlib,importlib.util,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;REV=BASE/'e1_independent_review'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def terminal(start,end):
 s=read(start);t=read(end)
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']!=t.get('worker_pid'):raise ValueError('supervised process not completed')
 return s
def resolved_command(s):
 cwd=Path(s['cwd']);return [x if x.startswith('-') else str((cwd/x).resolve()) if not Path(x).is_absolute() else str(Path(x).resolve()) for x in s['command']]
def plan():
 p=read(HERE/'execution_plan003.json')
 if p.get('stage')!='62-input four-budget fixed32 reference generation' or p.get('jobs')!=248:raise ValueError('wrong reference plan')
 for path,digest in p['source_sha256'].items():
  if sha(path)!=digest:raise ValueError('reference source changed '+path)
 return p
def prior():
 path=BASE/'e4_c_budget_ttft_pilot_v3/identity_gate.py';spec=importlib.util.spec_from_file_location('_bulk_reference_prior_identity',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m.admit_identity()
def execution_cpu(p):
 proof=read(HERE/'execution_cpu003.json');start=HERE/'monitor_execution_cpu003/started.json';end=HERE/'monitor_execution_cpu003/terminal.json';s=terminal(start,end)
 if resolved_command(s).count(str(HERE/'freeze_execution.py'))!=1 or proof.get('status')!='passed' or proof.get('test_returncode')!=0 or proof.get('GPU_authorized') is not False or proof.get('predictions_opened') is not True or proof.get('plan_sha256')!=sha(HERE/'execution_plan003.json') or proof.get('source_sha256')!=p['source_sha256']:raise ValueError('bulk CPU source proof mismatch')
 return proof
def admit_cpu_preflight(p):
 execution_cpu(p)
 proof=REV/'e4c_multibudget_references_worker_preflight_review.json';reviewer=REV/'review_e4c_multibudget_references_worker_preflight.py'
 if not proof.is_file():raise ValueError('CPU container locked: independent preflight review missing')
 r=read(proof);start=REV/'monitor_e4c_multibudget_references_worker_preflight/started.json';end=REV/'monitor_e4c_multibudget_references_worker_preflight/terminal.json';s=terminal(start,end);cmd=resolved_command(s)
 fields={'plan_sha256':HERE/'execution_plan003.json','execution_cpu_sha256':HERE/'execution_cpu003.json','execution_cpu_started_sha256':HERE/'monitor_execution_cpu003/started.json','execution_cpu_terminal_sha256':HERE/'monitor_execution_cpu003/terminal.json','reviewer_sha256':reviewer}
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('scope')!='references_cpu_preflight_only' or r.get('errors')!=[] or r.get('GT_opened') is not False or r.get('predictions_opened') is not True or r.get('scoring_performed') is not False or r.get('source_sha256')!=p['source_sha256'] or any(r.get(k)!=sha(v) for k,v in fields.items()) or cmd.count(str(reviewer))!=1 or cmd.count('--out')!=1 or cmd[cmd.index('--out')+1]!=str(proof):raise ValueError('reference CPU preflight review mismatch')
 return r
def admit_fixed_identity():
 # Fixed32 12-pair formal and independent reviews have not yet been produced.
 # Deliberately fail closed until their final, independently reviewed schema is
 # specified and actually exists; no code path should infer identity from CPU.
 raise ValueError('GPU locked: completed 12-pair fixed32 identity proof not available')
def admit_gpu():
 p=plan();prior();execution_cpu(p);fixed=admit_fixed_identity()
 raise ValueError('GPU locked: bulk reference independent prelaunch proof not available')
def check_supervisor(s,runid,now,p):
 progress=HERE/('run_'+runid)/'diagnostic/journal.jsonl';cmd=resolved_command(s)
 if (s.get('worker_pid')!=os.getpid() or s.get('process_group')!=os.getpgrp() or os.getpid()!=os.getpgrp() or s.get('timeout_seconds')!=p['outer_timeout_seconds'] or s.get('stale_seconds')!=p['stale_seconds'] or s.get('progress_paths')!=[str(progress)] or s.get('snapshot_seconds')!=3600 or s.get('poll_seconds')!=5 or s.get('grace_seconds')!=p['grace_seconds'] or cmd.count(str(HERE/'launch.py'))!=1 or s['command'].count('--run-id')!=1 or s['command'][s['command'].index('--run-id')+1]!=runid):raise ValueError('reference supervisor mismatch')
 initial=read(BASE/'initial_plan.json')
 if sha(BASE/'initial_plan.json')!=p['budget_source_sha256'] or now+p['outer_timeout_seconds']+p['acceptance_reserve_seconds']>initial['budget_end_unix'] or now<initial['budget_start_unix']:raise ValueError('reference deadline insufficient')
 return True
