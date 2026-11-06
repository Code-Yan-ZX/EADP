"""Machine gate for noGPU dependency probe; GPU remains unconditionally locked."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ended(s,t):
 a=read(s);b=read(t);return type(a.get('worker_pid')) is int and b.get('worker_pid')==a['worker_pid'] and b.get('reason')=='completed' and b.get('worker_returncode')==0 and b.get('remaining_process_group')==[]
def plan():
 p=read(HERE/'preflight_plan001.json')
 if p.get('scope')!='paired_timing_cpu_preflight_only' or p.get('GPU_authorized') is not False:raise ValueError('wrong plan')
 if any(sha(path)!=digest for path,digest in p['source_sha256'].items()):raise ValueError('frozen source changed')
 if not ended(HERE/'monitor_cpu004/started.json',HERE/'monitor_cpu004/terminal.json') or not ended(HERE/'monitor_references001/started.json',HERE/'monitor_references001/terminal.json'):raise ValueError('CPU proof incomplete')
 return p
def preflight():
 p=plan();r=ROOT/'e1_independent_review/rres_eadp_timing_v3_cpu_preflight_review.json';reviewer=ROOT/'e1_independent_review/review_rres_eadp_timing_v3_cpu_preflight.py';s=ROOT/'e1_independent_review/monitor_rres_eadp_timing_v3_cpu_preflight/started.json';t=s.parent/'terminal.json';q=read(r)
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('preflight_only') is not True or q.get('GPU_authorized') is not False or q.get('GT_opened') is not False or q.get('existing_predictions_opened') is not True or q.get('errors')!=[] or q.get('plan_sha256')!=sha(HERE/'preflight_plan001.json') or q.get('reviewer_sha256')!=sha(reviewer) or q.get('source_sha256')!=p['source_sha256'] or not ended(s,t):raise ValueError('missing independent noGPU review')
 return {'scope':p['scope'],'plan_sha256':sha(HERE/'preflight_plan001.json'),'review_sha256':sha(r)}
def gpu():raise RuntimeError('GPU not authorized')
