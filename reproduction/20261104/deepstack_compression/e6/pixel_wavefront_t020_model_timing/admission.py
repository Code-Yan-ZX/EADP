"""Fail-closed design gate; this module grants no real-model or GPU authority."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review';A=B/'e6/pixel_wavefront_t020_accept'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def supervised(directory,script,out=None):
 s=read(directory/'started.json');t=read(directory/'terminal.json');cwd=Path(s['cwd']);c=s['command']
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (cwd/c[1]).resolve()!=script.resolve() or (out is not None and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervisor incomplete or wrong command')
def require_identity():
 design=read(D/'design001.json')
 for name in ('contract','test'):
  filename='test_contract.py' if name=='test' else 'contract.py'
  if design[name+'_sha256']!=sha(D/filename):raise ValueError('design source changed')
 formal=A/'verification001.json';review=I/'pixel_wavefront_t020_run_result_review.json';reviewer=I/'review_pixel_wavefront_t020_run_result.py';r=read(review);f=read(formal)
 supervised(A/'monitor_verify001',A/'verify.py',formal);supervised(I/'monitor_pixel_wavefront_t020_run_result',reviewer,review)
 if (f.get('status')!='passed' or f.get('images')!=3 or f.get('jobs')!=6 or f.get('GT_opened') is not False or f.get('scoring_performed') is not False or f.get('plan_sha256')!=design['execution_plan_sha256'] or r.get('status')!='passed' or r.get('independent_review') is not True or r.get('GT_opened') is not False or r.get('scoring_performed') is not False or r.get('formal_sha256')!=sha(formal) or r.get('reviewer_sha256')!=sha(reviewer)):raise ValueError('identity proof mismatch')
 return {'formal_sha256':sha(formal),'review_sha256':sha(review),'design_sha256':sha(D/'design001.json')}
def authorize_gpu():raise RuntimeError('GPU timing not implemented or authorized')
