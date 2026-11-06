"""No-GPU processor preflight gate. Model and GPU are always locked."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if len(cmd)<2 or (cwd/cmd[1]).resolve()!=script.resolve() or s['worker_pid']!=t['worker_pid'] or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervision')
def admit_preflight():
 plan=read(D/'preflight_plan001.json');terminal(D/'monitor_cpu002',D/'cpu_check.py');c=read(D/'cpu_proof002.json')
 if plan.get('scope')!='three-anchor low-res processor noGPU only' or plan.get('GPU_authorized') is not False or c.get('status')!='passed' or c['source_sha256_before']!=c['source_sha256_after'] or any(sha(k)!=v for k,v in plan['source_sha256'].items()) or any(plan['source_sha256'].get(k)!=v for k,v in c['source_sha256_after'].items()):raise ValueError('source/CPU plan')
 qpath=I/'rres_k256_processor_v2_preflight_review.json';reviewer=I/'review_rres_k256_processor_v2_preflight.py';terminal(I/'monitor_rres_k256_processor_v2_preflight',reviewer,qpath);q=read(qpath)
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('preflight_only') is not True or q.get('GPU_authorized') is not False or q.get('GT_opened') is not False or q.get('errors')!=[] or q.get('source_sha256')!=plan['source_sha256'] or q.get('plan_sha256')!=sha(D/'preflight_plan001.json') or q.get('reviewer_sha256')!=sha(reviewer):raise ValueError('independent review')
 return {'plan_sha256':sha(D/'preflight_plan001.json'),'review_sha256':sha(qpath)}
def admit_gpu():raise RuntimeError('GPU locked: real model, DS, position, launcher and verifier not yet independently reviewed')
