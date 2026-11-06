"""Fail-closed CPU-only preflight gate; cannot authorize model or GPU."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];R=B/'e1_independent_review';PLAN=D/'runtime_plan003.json';REVIEW=R/'random_smoke_preflight_review_002.json';MON=R/'monitor_random_smoke_preflight_002';REVIEWER=R/'review_random_smoke_preflight_002.py'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[] or s['worker_pid']!=t['worker_pid'] or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()]:raise ValueError('review supervisor mismatch')
 if out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()):raise ValueError('review output mismatch')
def verify_plan():
 p=read(PLAN)
 if p['stage']!='preflight_only' or p['GPU_authorized'] is not False or p['GT_opened'] is not False or p['predictions_opened'] is not True or p['manifest_sha256']!=sha(D/'selection_manifest001.jsonl') or p['random_plan_sha256']!=sha(D/'plan001.json'):raise ValueError('preflight plan mismatch')
 for f,h in p['source_sha256'].items():
  if sha(f)!=h:raise ValueError('source changed: '+f)
 return p
def require_review():
 p=verify_plan();terminal(MON,REVIEWER,REVIEW);v=read(REVIEW)
 if v.get('status')!='passed' or v.get('independent_review') is not True or v.get('preflight_only') is not True or v.get('runtime_plan_sha256')!=sha(PLAN) or v.get('source_sha256')!=p['source_sha256'] or v.get('reviewer_sha256')!=sha(REVIEWER) or v.get('review_started_sha256')!=sha(MON/'started.json') or v.get('GT_opened') is not False or v.get('GPU_used') is not False or v.get('predictions_opened') is not True:raise ValueError('review not valid')
 return {'runtime_plan_sha256':sha(PLAN),'review_sha256':sha(REVIEW)}
def require_gpu():raise RuntimeError('GPU LOCKED: random smoke/full worker, real identity and final review absent')
