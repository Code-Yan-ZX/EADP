"""Draft fail-closed gates for a future random real-model smoke.

No approval proof exists. Both model and container remain locked until CPU
source freeze and independent reviews are completed.
"""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];R=B/'e1_independent_review'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def terminal(directory,reviewer,out):
 s=read(directory/'started.json');t=read(directory/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if (t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or
     [(cwd/x).resolve() for x in cmd if Path(x).name==reviewer.name]!=[reviewer.resolve()] or cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()):raise ValueError('review supervisor identity/exit')
def plan():
 path=D/'execution_plan009.json'
 if not path.is_file():raise ValueError('random model execution remains locked: plan missing')
 p=read(path)
 if p.get('scope')!='random K256 three-task 18-arm diagnostic' or p.get('GPU_authorized') is not False or p.get('GT_opened') is not False:raise ValueError('plan identity')
 for path,digest in p['source_sha256'].items():
  if sha(path)!=digest:raise ValueError('source changed '+path)
 return p
def cpu(p):
 proof=D/'model_cpu_proof009.json';started=D/'monitor_model_cpu009/started.json';ended=D/'monitor_model_cpu009/terminal.json'
 v=read(proof);s=read(started);t=read(ended);cwd=Path(s['cwd']);cmd=s['command']
 if (v.get('status')!='passed' or v.get('test_returncode')!=0 or v.get('model_loaded') is not False or v.get('GPU_authorized') is not False or v.get('GT_opened') is not False or v.get('predictions_opened') is not True or v.get('source_sha256_before')!=v.get('source_sha256_after') or
     t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s.get('worker_pid')!=t.get('worker_pid') or cwd.resolve()!=B.resolve() or
     [(cwd/x).resolve() for x in cmd if Path(x).name=='freeze_model_execution.py']!=[(D/'freeze_model_execution.py').resolve()]):raise ValueError('random CPU proof/supervisor mismatch')
 expected=dict(v['source_sha256_after']);expected[str(proof.resolve())]=sha(proof)
 if p.get('source_sha256')!=expected:raise ValueError('random CPU source inventory mismatch')
 return {'cpu_proof_sha256':sha(proof),'cpu_started_sha256':sha(started),'cpu_terminal_sha256':sha(ended)}
def admit(preflight):
 p=plan();cpu_fields=cpu(p);kind='preflight' if preflight else 'final';proof=R/('random_model_'+kind+'_review.json');reviewer=R/('review_random_model_'+kind+'.py');monitor=R/('monitor_random_model_'+kind)
 if not proof.is_file():raise ValueError('random model '+kind+' independent review missing')
 terminal(monitor,reviewer,proof);v=read(proof)
 if (v.get('status')!='passed' or v.get('independent_review') is not True or v.get('scope')!=('noGPU preflight only' if preflight else 'GPU diagnostic prelaunch') or v.get('errors')!=[] or v.get('GT_opened') is not False or v.get('scoring_performed') is not False or v.get('predictions_opened') is not True or v.get('plan_sha256')!=sha(D/'execution_plan009.json') or v.get('reviewer_sha256')!=sha(reviewer) or v.get('source_sha256')!=p['source_sha256'] or any(v.get(k)!=h for k,h in cpu_fields.items())):raise ValueError('random model review mismatch')
 if not preflight:
  pre=D/'model_preflight002';start=D/'monitor_model_preflight002/started.json';end=D/'monitor_model_preflight002/terminal.json'
  for k,path in (('preflight_started_sha256',start),('preflight_terminal_sha256',end),('worker_preflight_sha256',pre/'worker_preflight.json'),('preflight_launch_sha256',pre/'launch.json'),('preflight_exit_sha256',pre/'exit.json'),('preflight_review_sha256',R/'random_model_preflight_review.json')):
   if v.get(k)!=sha(path):raise ValueError('random worker preflight evidence mismatch '+k)
  s=read(start);t=read(end);ex=read(pre/'exit.json');w=read(pre/'worker_preflight.json');cmd=s['command'];cwd=Path(s['cwd'])
  if [(cwd/x).resolve() for x in cmd if Path(x).name=='model_launch.py']!=[(D/'model_launch.py').resolve()] or cmd.count('--preflight-only')!=1 or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s['worker_pid']!=t.get('worker_pid') or w.get('status')!='passed' or w.get('plan_sha256')!=sha(D/'execution_plan009.json') or w.get('anchors')!=3 or w.get('random_choices')!=9 or w.get('GT_opened') is not False or w.get('predictions_opened') is not True or w.get('cuda_initialized') is not False or w.get('model_loaded') is not False or ex.get('returncode')!=0 or ex.get('container_exit_code')!=0 or ex.get('container_stopped_verified') is not True or ex.get('timed_out') is not False or ex.get('interrupted') is not None or (ex.get('last_inspect') or {}).get('running') is not False or (ex.get('last_inspect') or {}).get('pid')!=0 or (ex.get('last_inspect') or {}).get('host_devices') or (ex.get('last_inspect') or {}).get('device_requests'):raise ValueError('random worker preflight incomplete')
 return {'plan_sha256':sha(D/'execution_plan009.json'),'review_sha256':sha(proof),'source_sha256':p['source_sha256']}
