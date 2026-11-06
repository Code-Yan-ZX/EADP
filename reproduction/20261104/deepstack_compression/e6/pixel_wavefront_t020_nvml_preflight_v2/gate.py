"""Only allow independently reviewed no-GPU dependency preflight."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if len(c)<2 or (cwd/c[1]).resolve()!=script.resolve() or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or (out is not None and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervisor')
def admit():
 plan=read(D/'plan001.json');qpath=I/'pixel_t020_nvml_preflight_v2_review.json';reviewer=I/'review_pixel_t020_nvml_preflight_v2.py';monitor=I/'monitor_pixel_t020_nvml_preflight_v2';q=read(qpath);terminal(monitor,reviewer,qpath)
 if plan.get('scope')!='pixel nine-reference noGPU imports only' or plan.get('GPU_authorized') is not False or any(sha(k)!=v for k,v in plan['source_sha256'].items()):raise ValueError('plan/source')
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('preflight_only') is not True or q.get('GPU_authorized') is not False or q.get('GT_opened') is not False or q.get('existing_DEV_predictions_opened') is not True or q.get('errors')!=[] or q.get('plan_sha256')!=sha(D/'plan001.json') or q.get('reviewer_sha256')!=sha(reviewer) or q.get('source_sha256')!=plan['source_sha256']:raise ValueError('independent review')
 return {'plan_sha256':sha(D/'plan001.json'),'review_sha256':sha(qpath)}
def gpu():raise RuntimeError('GPU locked: Pixel NVML execution, sampler and formal acceptance are not yet reviewed')
