"""Exploratory 32-image performance pilot is intentionally still locked."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;REV=BASE/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def plan():
 p=read(HERE/'runtime_plan001.json')
 if p.get('stage')!='post-hoc 32-image 4-budget natural/fixed32 timing pilot' or p.get('jobs')!=1520:raise ValueError('invalid timing plan')
 for path,digest in p['source_sha256'].items():
  if sha(path)!=digest:raise ValueError('timing source changed '+path)
 return p
def admit_gpu():
 # This is deliberately unconditional until both independent fixed32 identity
 # and all-248-reference formal reviews exist and a future gate is reviewed.
 raise ValueError('GPU locked: fixed32 identity, 248 references and timing prelaunch not accepted')

def terminal(start,end):
 s=read(start);t=read(end)
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']!=t.get('worker_pid'):raise ValueError('supervised process not completed')
 return s
def resolved_command(s):
 cwd=Path(s['cwd']);return [x if x.startswith('-') else str((cwd/x).resolve()) if not Path(x).is_absolute() else str(Path(x).resolve()) for x in s['command']]
def admit_cpu_preflight():
 p=plan();cpu=read(HERE/'execution_cpu001.json');cs=HERE/'monitor_execution_cpu001/started.json';ct=HERE/'monitor_execution_cpu001/terminal.json';s=terminal(cs,ct)
 if resolved_command(s).count(str(HERE/'freeze_execution.py'))!=1 or cpu.get('status')!='passed' or cpu.get('test_returncode')!=0 or cpu.get('predictions_opened') is not True or cpu.get('model_loaded') is not False or cpu.get('GPU_authorized') is not False or cpu.get('plan_sha256')!=sha(HERE/'runtime_plan001.json') or cpu.get('source_sha256')!=p['source_sha256']:raise ValueError('timing CPU proof mismatch')
 proof=REV/'e4c_multibudget_worker_preflight_review.json';reviewer=REV/'review_e4c_multibudget_worker_preflight.py'
 if not proof.is_file():raise ValueError('CPU container locked: independent source review unavailable')
 r=read(proof);ms=REV/'monitor_e4c_multibudget_worker_preflight/started.json';mt=REV/'monitor_e4c_multibudget_worker_preflight/terminal.json';s=terminal(ms,mt);cmd=resolved_command(s)
 fields={'plan_sha256':HERE/'runtime_plan001.json','execution_cpu_sha256':HERE/'execution_cpu001.json','execution_cpu_started_sha256':cs,'execution_cpu_terminal_sha256':ct,'reviewer_sha256':reviewer}
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('scope')!='multibudget_worker_cpu_preflight_only' or r.get('errors')!=[] or r.get('GT_opened') is not False or r.get('predictions_opened') is not True or r.get('scoring_performed') is not False or r.get('source_sha256')!=p['source_sha256'] or any(r.get(k)!=sha(v) for k,v in fields.items()) or cmd.count(str(reviewer))!=1 or cmd.count('--out')!=1 or cmd[cmd.index('--out')+1]!=str(proof):raise ValueError('timing CPU review mismatch')
 return r
