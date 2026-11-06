"""Fail-closed source and proof checks; no GPU operations."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;V2=BASE/'e4_c_budget_pilot_v2';REV=BASE/'e1_independent_review'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for part in iter(lambda:f.read(1048576),b''):h.update(part)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def terminal(start,end):
 s=read(start);t=read(end)
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']!=t.get('worker_pid'):raise ValueError('supervisor did not complete')

def validate_prior():
 proof=REV/'e4c_budget_pilot_v2_preflight_review.json';reviewer=REV/'review_e4c_budget_pilot_v2_preflight.py'
 start=REV/'monitor_e4c_budget_pilot_v2_preflight/started.json';end=REV/'monitor_e4c_budget_pilot_v2_preflight/terminal.json'
 plan=V2/'preflight_plan001.json';result=V2/'preflight001.json';ps=V2/'monitor_preflight001/started.json';pt=V2/'monitor_preflight001/terminal.json';r=read(proof);p=read(plan)
 if (r.get('status')!='passed' or r.get('independent_review') is not True or r.get('errors')!=[] or r.get('GT_opened') is not False or
     r.get('scoring_performed') is not False or r.get('gpu_authorized') is not False or r.get('source_sha256')!=p['source_sha256'] or
     r.get('plan_sha256')!=sha(plan) or r.get('preflight_sha256')!=sha(result) or r.get('preflight_started_sha256')!=sha(ps) or
     r.get('preflight_terminal_sha256')!=sha(pt) or r.get('reviewer_sha256')!=sha(reviewer)):raise ValueError('prior independent proof invalid')
 terminal(start,end);terminal(ps,pt)
 if read(result).get('status')!='passed' or read(result).get('cuda_initialized') is not False:raise ValueError('prior worker preflight failed')
 for path,digest in p['source_sha256'].items():
  if sha(path)!=digest:raise ValueError('prior source changed '+path)
 return {'prior_plan_sha256':sha(plan),'prior_proof_sha256':sha(proof),'prior_review_terminal_sha256':sha(end),'source_sha256':p['source_sha256']}

def validate_runtime_sources():
 plan=read(HERE/'runtime_plan001.json')
 for path,digest in plan['source_sha256'].items():
  if sha(path)!=digest:raise ValueError('runtime source changed '+path)
 return plan

def admit_gpu():
 # A real GPU launcher, source-pinned final reviewer and the latter's proof
 # are deliberately not implemented. No file or user-supplied flag bypasses
 # this lock. A future version must add and re-review the complete chain.
 raise RuntimeError('GPU locked: bounded controller and final independent prelaunch proof absent')
