"""B proof validation; GPU stays locked until lightweight endpoint proof exists."""
import hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def supervised(start_path,terminal_path,script,output):
 started=read(start_path);terminal=read(terminal_path);cmd=started.get('command',[]);cwd=Path(started.get('cwd','/'))
 if (terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[] or
     type(started.get('worker_pid')) is not int or started['worker_pid']!=terminal.get('worker_pid') or
     [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or cmd.count('--out')!=1 or
     cmd.index('--out')+1>=len(cmd) or (cwd/cmd[cmd.index('--out')+1]).resolve()!=output.resolve()):raise ValueError('B verifier supervisor failed')

def validate_b_identity(base=BASE):
 """Validate both B proofs and evidence; does not admit lightweight timing."""
 b=base/'e4_b_identity';review_dir=base/'e1_independent_review'
 formal=b/'b_001_verification.json';fs=b/'monitor_verify_b001/started.json';ft=b/'monitor_verify_b001/terminal.json';verifier=b/'b_verify.py'
 review=review_dir/'b001_review.json';rs=review_dir/'monitor_b001/started.json';rt=review_dir/'monitor_b001/terminal.json';reviewer=review_dir/'review_b001.py'
 supervised(fs,ft,verifier,formal);supervised(rs,rt,reviewer,review)
 f=read(formal);r=read(review)
 if (f.get('status')!='passed' or f.get('stage')!='skip_drop_identity' or f.get('natural_pairs')!=3 or f.get('forced_pairs')!=1 or f.get('generations')!=8 or
     f.get('GT_opened') is not False or f.get('scoring_performed') is not False or f.get('timing_claims') is not False or f.get('verifier_sha256')!=sha(verifier) or
     r.get('status')!='passed' or r.get('independent_review') is not True or r.get('n_generations')!=8 or r.get('natural_pairs')!=3 or r.get('forced_pairs')!=1 or
     r.get('GT_opened') is not False or r.get('scoring_performed') is not False or r.get('timing_claims') is not False or
     r.get('formal_proof_sha256')!=sha(formal) or r.get('formal_started_sha256')!=sha(fs) or r.get('formal_terminal_sha256')!=sha(ft) or r.get('reviewer_sha256')!=sha(reviewer)):
  raise ValueError('B identity proof mismatch')
 files=f.get('files_sha256');arrays=f.get('array_sha256')
 if not isinstance(files,dict) or not isinstance(arrays,dict) or len(arrays)!=8 or r.get('files_sha256')!=files or r.get('array_sha256')!=arrays:raise ValueError('B evidence inventory mismatch')
 required=[b/'b_001'/x for x in ('launch.json','exit.json','frozen.json','worker_complete.json','records.jsonl','generated.jsonl','comparisons.jsonl')]
 for p in required:
  if files.get(str(p.resolve()))!=sha(p):raise ValueError('B required evidence missing')
 for p,digest in [*files.items(),*arrays.items()]:
  if sha(p)!=digest:raise ValueError('B evidence changed')
 return {'formal_proof_sha256':sha(formal),'review_proof_sha256':sha(review),'formal_terminal_sha256':sha(ft),'review_terminal_sha256':sha(rt),
         'scope':'three DEV natural and one forced DROP/SKIP pairs; not lightweight adapter identity'}

def require_gpu_identity(*_args,**_kwargs):
 # Deliberately non-configurable until independently accepted real model
 # lightweight endpoint and verified launcher/device proofs exist.
 raise ValueError('E4-C real timing locked: C-SKIP identity and lightweight adapter endpoint proof not yet independently accepted')
