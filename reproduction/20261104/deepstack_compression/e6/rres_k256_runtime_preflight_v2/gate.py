"""Only release a no-device runtime-import probe; never grants GPU."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review';P=B/'e6/rres_k256_pilot_v2'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if s['worker_pid']!=t['worker_pid'] or t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[] or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervision')
def admit():
 plan=read(D/'plan001.json');terminal(D/'monitor_cpu002',D/'cpu_check.py');cpu=read(D/'cpu_proof002.json')
 if plan.get('scope')!='runtime noGPU import only' or plan.get('GPU_authorized') is not False or cpu.get('status')!='passed' or cpu['before']!=cpu['after'] or any(sha(k)!=v for k,v in plan['source_sha256'].items()):raise ValueError('plan/CPU')
 old=I/'rres_k256_processor_v2_preflight_result_review.json';oldscript=I/'review_rres_k256_processor_v2_preflight_result.py';terminal(I/'monitor_rres_k256_processor_v2_preflight_result',oldscript,old);o=read(old)
 if o.get('status')!='passed' or o.get('independent_review') is not True or o.get('worker_sha256')!=sha(P/'run_preflight001/worker_preflight.json') or o.get('reviewer_sha256')!=sha(oldscript) or o.get('images')!=3 or o.get('GPU_authorized') is not False or o.get('errors')!=[]:raise ValueError('prior preflight')
 out=I/'rres_k256_runtime_v2_preflight_review.json';reviewer=I/'review_rres_k256_runtime_v2_preflight.py';terminal(I/'monitor_rres_k256_runtime_v2_preflight',reviewer,out);r=read(out)
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('GPU_authorized') is not False or r.get('GT_opened') is not False or r.get('errors')!=[] or r.get('plan_sha256')!=sha(D/'plan001.json') or r.get('source_sha256')!=plan['source_sha256'] or r.get('reviewer_sha256')!=sha(reviewer):raise ValueError('independent review')
 return {'plan_sha256':sha(D/'plan001.json'),'review_sha256':sha(out)}
