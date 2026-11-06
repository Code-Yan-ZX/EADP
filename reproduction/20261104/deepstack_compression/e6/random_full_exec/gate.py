"""Fail-closed source and independent-review admission; no implicit authority."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review';S=B/'e6/random_smoke';A=B/'e6/random_smoke_accept'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(p):
 t=read(p)
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[]:raise ValueError('supervised process failed '+str(p))
 return t
def supervised(directory,script):
 t=terminal(directory/'terminal.json');s=read(directory/'started.json');cwd=Path(s['cwd'])
 if s.get('worker_pid')!=t.get('worker_pid') or sum((cwd/x).resolve()==Path(script).resolve() for x in s['command'])!=1:raise ValueError('supervisor command/identity')
def plan():
 p=read(D/'runtime_plan001.json')
 if p.get('stage')!='random_full_dev' or p.get('questions')!=384 or p.get('generations')!=1152 or p.get('GT_opened') is not False:raise ValueError('runtime plan')
 for name,digest in p['source_sha256'].items():
  if sha(name)!=digest:raise ValueError('source changed '+name)
 return p
def prior():
 m=read(S/'manifest_validation001.json')
 if (m.get('status')!='passed' or m.get('questions')!=384 or m.get('selections')!=1152 or m.get('GT_opened') is not False or m.get('selection_manifest_sha256')!=sha(S/'selection_manifest001.jsonl')):raise ValueError('selection audit')
 supervised(S/'monitor_validate001',S/'validate_manifest.py');supervised(A/'monitor_verify001',A/'verify.py');supervised(I/'monitor_random_smoke_run001',I/'review_random_smoke_run001.py')
 formal=read(A/'verification001.json');review=read(I/'random_smoke_run001_review.json')
 if formal.get('status')!='passed' or formal.get('arms')!=18 or formal.get('GT_opened') is not False or formal.get('errors')!=[]:raise ValueError('smoke formal')
 if m.get('reviewer_sha256')!=sha(S/'validate_manifest.py') or formal.get('verifier_sha256')!=sha(A/'verify.py') or review.get('reviewer_sha256')!=sha(I/'review_random_smoke_run001.py'):raise ValueError('prior reviewer source')
 if (review.get('status')!='passed' or review.get('independent_review') is not True or review.get('arms')!=18 or review.get('formal_sha256')!=sha(A/'verification001.json') or review.get('formal_started_sha256')!=sha(A/'monitor_verify001/started.json') or review.get('formal_terminal_sha256')!=sha(A/'monitor_verify001/terminal.json') or review.get('GT_opened') is not False):raise ValueError('smoke independent')
 for path,digest in {**formal['files_sha256'],**formal['array_sha256']}.items():
  if sha(path)!=digest:raise ValueError('smoke evidence changed')
 return {'manifest_validation_sha256':sha(S/'manifest_validation001.json'),'smoke_formal_sha256':sha(A/'verification001.json'),'smoke_review_sha256':sha(I/'random_smoke_run001_review.json')}
def review(proof,monitor,reviewer,expected):
 terminal(monitor/'terminal.json');v=read(proof);started=read(monitor/'started.json')
 if v.get('status')!='passed' or v.get('independent_review') is not True or v.get('runtime_plan_sha256')!=sha(D/'runtime_plan001.json') or v.get('scope')!=expected or v.get('errors')!=[]:raise ValueError('independent review absent or incorrect')
 cwd=Path(started['cwd'])
 if v.get('reviewer_sha256')!=sha(reviewer) or sum((cwd/x).resolve()==reviewer.resolve() for x in started['command'])!=1:raise ValueError('reviewer identity')
 return sha(proof)
def admit(cpu):
 p=plan();old=prior()
 preliminary=review(I/'random_full_worker_preflight_review.json',I/'monitor_random_full_worker_preflight',I/'review_random_full_worker_preflight.py','CPU worker preflight only')
 approval={'plan_sha256':sha(D/'runtime_plan001.json'),'prior':old,'preflight_review_sha256':preliminary}
 if not cpu:
  v=read(D/'run_preflight001/worker_preflight.json');terminal(D/'monitor_preflight001/terminal.json')
  if v.get('status')!='passed' or v.get('questions')!=384 or v.get('choices')!=1152 or v.get('cuda_initialized') is not False or v.get('GT_opened') is not False:raise ValueError('complete no-GPU preflight missing')
  final=read(I/'random_full_prelaunch_review.json')
  if final.get('preflight_sha256')!=sha(D/'run_preflight001/worker_preflight.json') or final.get('preflight_terminal_sha256')!=sha(D/'monitor_preflight001/terminal.json') or final.get('verification_plan_sha256')!=sha(D/'verification_plan001.json'):raise ValueError('preflight/verification proof binding')
  approval['final_review_sha256']=review(I/'random_full_prelaunch_review.json',I/'monitor_random_full_prelaunch',I/'review_random_full_prelaunch.py','GPU full DEV only')
 return approval
