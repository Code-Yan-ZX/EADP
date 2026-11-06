"""Planning gate only. A real GPU launcher does not yet exist and is locked."""
import hashlib,json,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;DEADLINE=B/'initial_plan.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def check_plan(plan,now,outer_seconds,acceptance_seconds):
 if type(outer_seconds) is not int or outer_seconds<=0 or type(acceptance_seconds) is not int or acceptance_seconds<=0:raise ValueError('invalid timing budget')
 if sha(DEADLINE)!=plan['budget_source_sha256']:raise ValueError('deadline source changed')
 d=json.loads(DEADLINE.read_text())
 if now+outer_seconds+acceptance_seconds>d['budget_end_unix']:raise ValueError('three-day budget insufficient')
 for p,h in plan['source_sha256'].items():
  if sha(p)!=h:raise ValueError('frozen plan input changed: '+p)
 if plan.get('GPU_authorized') is not False or plan.get('measured_jobs')!=1280 or plan.get('warmup_jobs')!=240:raise ValueError('plan contract mismatch')
 return True
def require_gpu(*_args,**_kwargs):
 # Placeholder is deliberately not a launcher: real worker/adapter, fixed32
 # reference checks, CPU/proof/container chain and root scheduling are missing.
 raise RuntimeError('GPU LOCKED: real pilot worker and independently reviewed admission are not implemented')
