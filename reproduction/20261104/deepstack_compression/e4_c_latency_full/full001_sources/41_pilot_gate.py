"""Accepted 72-job pilot evidence, independent reviewer and fixed budget."""
import json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;PILOT=BASE/'e4_c_latency';REVIEW=BASE/'e1_independent_review'
sys.path.insert(0,str(PILOT))
from identity_gate import sha,supervised
from latency_gate import require_light_identity

F=PILOT/'latency_001_verification.json';FS=PILOT/'monitor_verify_latency001/started.json';FT=PILOT/'monitor_verify_latency001/terminal.json'
R=REVIEW/'e4c_latency001_review.json';RS=REVIEW/'monitor_e4c_latency001/started.json';RT=REVIEW/'monitor_e4c_latency001/terminal.json';REVIEWER=REVIEW/'review_e4c_latency001.py'
RUN=PILOT/'latency_001';RUNS=PILOT/'monitor_latency001/started.json';RUNT=PILOT/'monitor_latency001/terminal.json';PLAN=PILOT/'latency_launch_plan001.json'
ANNOT=PILOT/'latency_001_annotations.jsonl';HEALTH=PILOT/'monitor_latency001/health.jsonl';BUDGET=HERE/'budget_plan.json'

def pilot_gate(*,host=True):
 for p in (F,FS,FT,R,RS,RT,RUNS,RUNT,PLAN,ANNOT,HEALTH):
  if not p.is_file():raise ValueError('full timing locked: pilot evidence missing')
 supervised(FS,FT,PILOT/'latency_verify.py',F);supervised(RS,RT,REVIEWER,R)
 f=json.loads(F.read_text());r=json.loads(R.read_text());plan=json.loads(PLAN.read_text());budget=json.loads(BUDGET.read_text())
 if (f.get('status')!='passed' or f.get('stage')!='smoke' or f.get('n_jobs')!=72 or f.get('n_generated')!=72 or f.get('errors')!=[] or f.get('GT_opened') is not False or f.get('scoring_performed') is not False or f.get('verifier_sha256')!=sha(PILOT/'latency_verify.py')):raise ValueError('pilot formal proof mismatch')
 if (r.get('status')!='passed' or r.get('independent_review') is not True or r.get('n_jobs')!=72 or r.get('n_generated')!=72 or r.get('errors')!=[] or r.get('GT_opened') is not False or r.get('scoring_performed') is not False or
     r.get('formal_proof_sha256')!=sha(F) or r.get('formal_started_sha256')!=sha(FS) or r.get('formal_terminal_sha256')!=sha(FT) or r.get('run_started_sha256')!=sha(RUNS) or r.get('run_terminal_sha256')!=sha(RUNT) or
     r.get('launch_plan_sha256')!=sha(PLAN) or r.get('health_sha256')!=sha(HEALTH) or r.get('annotations_sha256')!=sha(ANNOT) or r.get('reviewer_sha256')!=sha(REVIEWER)):raise ValueError('pilot independent review mismatch')
 if budget.get('pilot_formal_sha256')!=sha(F) or budget.get('pilot_independent_sha256')!=sha(R):raise ValueError('budget does not bind accepted pilot')
 ev=f.get('evidence_sha256',{});required=[RUN/x for x in ('launch.json','exit.json','frozen.json','worker_complete.json','journal.jsonl')]+[RUNS,RUNT,HEALTH,ANNOT]
 if any(ev.get(str(p.resolve()))!=sha(p) for p in required) or any(sha(p)!=d for p,d in ev.items()):raise ValueError('pilot evidence changed')
 source=f.get('source_sha256',{});snap=plan.get('source_snapshots',{})
 if source!=plan.get('approval_bundle',{}).get('source_sha256') or set(source)!=set(snap):raise ValueError('pilot source inventory changed')
 for p,d in source.items():
  if sha(p)!=d or snap[p].get('sha256')!=d or sha(snap[p]['snapshot'])!=d:raise ValueError('pilot source changed')
 terminal=json.loads(RUNT.read_text());launch=json.loads((RUN/'launch.json').read_text());ex=json.loads((RUN/'exit.json').read_text());ident=ex.get('container_identity') or {}
 if (terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[] or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or ex.get('gpu_pids_after')!=[]):raise ValueError('pilot completion mismatch')
 if host:
  obj=json.loads(subprocess.run(['docker','inspect',launch['name']],capture_output=True,text=True,timeout=15,check=True).stdout)[0]
  if obj['Id']!=ident.get('container_id') or obj['State']['Running'] is not False or obj['State']['Pid']!=0 or obj['State']['ExitCode']!=0:raise ValueError('pilot actual container mismatch')
 return {'formal_sha256':sha(F),'independent_sha256':sha(R),'run_started_sha256':sha(RUNS),'run_terminal_sha256':sha(RUNT),'launch_plan_sha256':sha(PLAN),'health_sha256':sha(HEALTH),'annotations_sha256':sha(ANNOT),'light_identity':require_light_identity()}
