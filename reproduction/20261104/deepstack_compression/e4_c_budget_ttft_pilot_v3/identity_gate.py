"""Fail closed on absent formal and independent 4-budget identity proof."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;FORMAL=BASE/'e4_c_budget_accept/verification002.json';REVIEW=BASE/'e1_independent_review/e4c_budget_identity001_review.json'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def terminal(start,end):
 s=read(start);t=read(end)
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']!=t.get('worker_pid'):raise ValueError('identity supervisor incomplete')
 return s
def admit_identity():
 if not FORMAL.is_file() or not REVIEW.is_file():raise ValueError('four-budget identity not independently accepted')
 f=read(FORMAL);r=read(REVIEW);fs=BASE/'e4_c_budget_accept/monitor_verify002/started.json';ft=BASE/'e4_c_budget_accept/monitor_verify002/terminal.json'
 rs=BASE/'e1_independent_review/monitor_e4c_budget_identity001/started.json';rt=BASE/'e1_independent_review/monitor_e4c_budget_identity001/terminal.json'
 reviewer=BASE/'e1_independent_review/review_e4c_budget_identity001.py';run=BASE/'e4_c_budget_execution';snap=run/'run_001_sources/manifest.json'
 if (f.get('status')!='passed' or f.get('pairs')!=16 or f.get('generations')!=32 or f.get('natural_pairs')!=12 or f.get('forced_pairs')!=4 or f.get('errors')!=[] or
     f.get('GT_opened') is not False or f.get('scoring_performed') is not False or f.get('timing_claims') is not False or
     r.get('status')!='passed' or r.get('independent_review') is not True or r.get('pairs')!=16 or r.get('generations')!=32 or r.get('errors')!=[] or
     r.get('GT_opened') is not False or r.get('scoring_performed') is not False or r.get('formal_sha256')!=sha(FORMAL) or
     r.get('formal_started_sha256')!=sha(fs) or r.get('formal_terminal_sha256')!=sha(ft) or r.get('reviewer_sha256')!=sha(reviewer) or
     r.get('run_started_sha256')!=sha(run/'monitor_run001/started.json') or r.get('run_terminal_sha256')!=sha(run/'monitor_run001/terminal.json') or
     r.get('snapshot_manifest_sha256')!=sha(snap) or f.get('snapshot_manifest_sha256')!=sha(snap)):raise ValueError('identity proof mismatch')
 terminal(fs,ft);s=terminal(rs,rt);cmd=s['command'];cwd=Path(s['cwd'])
 paths=[str((cwd/x).resolve()) if not Path(x).is_absolute() else str(Path(x).resolve()) for x in cmd]
 if paths.count(str(reviewer))!=1 or cmd.count('--out')!=1 or paths[cmd.index('--out')+1]!=str(REVIEW):raise ValueError('identity reviewer process mismatch')
 evidence={'plan_sha256':run/'execution_plan001.json','run_started_sha256':run/'monitor_run001/started.json','run_terminal_sha256':run/'monitor_run001/terminal.json',
           'launch_sha256':run/'run_001/launch.json','exit_sha256':run/'run_001/exit.json','worker_complete_sha256':run/'run_001/worker_complete.json',
           'journal_sha256':run/'run_001/diagnostic/journal.jsonl','verifier_sha256':BASE/'e4_c_budget_accept/verify.py',
           'verification_plan_sha256':BASE/'e4_c_budget_accept/verification_plan003.json',
           'snapshot_started_sha256':BASE/'e4_c_budget_accept/monitor_snapshot001/started.json','snapshot_terminal_sha256':BASE/'e4_c_budget_accept/monitor_snapshot001/terminal.json'}
 if any(f.get(k)!=sha(v) for k,v in evidence.items()):raise ValueError('identity evidence changed')
 terminal(BASE/'e4_c_budget_accept/monitor_snapshot001/started.json',BASE/'e4_c_budget_accept/monitor_snapshot001/terminal.json')
 arrays=f.get('array_sha256');root=(run/'run_001/diagnostic/arrays').resolve()
 if not isinstance(arrays,dict) or len(arrays)!=32 or r.get('array_sha256')!=arrays:raise ValueError('identity array inventory mismatch')
 for path,digest in arrays.items():
  if not Path(path).resolve().is_relative_to(root) or sha(path)!=digest:raise ValueError('identity array changed')
 snapshot=read(snap);sources=read(run/'execution_plan001.json')['source_sha256'];copies=snapshot.get('sources',{})
 if len(sources)!=142 or set(copies)!=set(sources):raise ValueError('identity source snapshot inventory mismatch')
 for original,digest in sources.items():
  item=copies[original];path=Path(item['snapshot']).resolve()
  if item.get('source')!=original or item.get('sha256')!=digest or not path.is_relative_to(snap.parent.resolve()) or sha(path)!=digest:raise ValueError('identity source snapshot changed')
 # No accepted proof alone authorizes the performance worker or GPU.
 return {'formal_sha256':sha(FORMAL),'review_sha256':sha(REVIEW),'snapshot_manifest_sha256':sha(snap)}
