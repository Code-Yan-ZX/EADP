"""CPU-only source and independent review gate; never authorizes model/GPU."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('CPU gate supervisor identity')
 return s
def plan():
 p=read(D/'cpu_plan001.json')
 if p.get('status')!='prepared_cpu_preflight' or p.get('GPU_authorized') is not False or p.get('GT_opened') is not False or p.get('target_predictions_opened') is not False or p.get('inputs')!=222 or p.get('jobs')!=7920:raise ValueError('bad CPU plan')
 if not p.get('source_sha256') or any(sha(k)!=v for k,v in p['source_sha256'].items()):raise ValueError('CPU source changed')
 return p
def admit():
 p=plan();cpu=read(D/'cpu_proof001.json');terminal(D/'monitor_cpu001',D/'freeze_cpu.py')
 if cpu.get('status')!='passed' or cpu.get('test_returncode')!=0 or cpu.get('plan_sha256')!=sha(D/'cpu_plan001.json') or cpu.get('source_sha256')!=p['source_sha256'] or cpu.get('GT_opened') is not False or cpu.get('target_predictions_opened') is not False or cpu.get('GPU_authorized') is not False:raise ValueError('CPU proof changed')
 proof=I/'e4c_worker192_v4_cpu_preflight_review.json';reviewer=I/'review_e4c_worker192_v4_cpu_preflight.py';mon=I/'monitor_e4c_worker192_v4_cpu_preflight'
 if not proof.is_file():raise ValueError('CPU container locked: independent source review missing')
 terminal(mon,reviewer,proof);r=read(proof);fields={'plan_sha256':D/'cpu_plan001.json','cpu_proof_sha256':D/'cpu_proof001.json','cpu_started_sha256':D/'monitor_cpu001/started.json','cpu_terminal_sha256':D/'monitor_cpu001/terminal.json','reviewer_sha256':reviewer}
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('scope')!='worker192_cpu_preflight_only' or r.get('errors')!=[] or r.get('GT_opened') is not False or r.get('target_predictions_opened') is not False or r.get('GPU_authorized') is not False or r.get('source_sha256')!=p['source_sha256'] or any(r.get(k)!=sha(v) for k,v in fields.items()):raise ValueError('independent CPU proof mismatch')
 return {'status':'passed','scope':'CPU worker preflight only','plan_sha256':sha(D/'cpu_plan001.json'),'review_sha256':sha(proof)}
