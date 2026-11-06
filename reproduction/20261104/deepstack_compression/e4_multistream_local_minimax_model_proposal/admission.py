"""Candidate fail-closed host/worker gate; a plan/reviews must be frozen separately."""
import hashlib,json,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;I=B/'e1_independent_review';END=1794065414

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def term(directory,script,out):
 s=read(directory/'started.json');t=read(directory/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if (s.get('worker_pid')!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or
     [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()):raise ValueError('review supervisor')
def review(name,scope,plan):
 proof=I/(name+'_review.json');reviewer=I/('review_'+name+'.py');monitor=I/('monitor_'+name)
 term(monitor,reviewer,proof);q=read(proof)
 if (q.get('status')!='passed' or q.get('independent_review') is not True or q.get('scope')!=scope or q.get('GT_opened') is not False or q.get('errors')!=[] or q.get('plan_sha256')!=sha(D/'execution_plan001.json') or q.get('reviewer_sha256')!=sha(reviewer)):
  raise ValueError('review proof')
 return {'proof_sha256':sha(proof),'reviewer_sha256':sha(reviewer),'started_sha256':sha(monitor/'started.json'),'terminal_sha256':sha(monitor/'terminal.json')}
def admit(preflight):
 p=read(D/'execution_plan001.json')
 if (p.get('status')!='frozen' or p.get('questions')!=32 or p.get('generations')!=64 or
     p.get('budget_end_unix')!=END or read(B/'initial_plan.json').get('budget_end_unix')!=END or
     time.time()+p['outer_timeout_seconds']+p['acceptance_reserve_seconds']>END or
     any(sha(path)!=digest for path,digest in p['source_sha256'].items())):raise ValueError('plan/source/deadline')
 prior=review('local_minimax_model32_cpu_preflight','cpu_preflight_only',p)
 if preflight:return {'scope':'cpu_preflight_only','plan_sha256':sha(D/'execution_plan001.json'),'review':prior}
 # No GPU execution until actual CPU preflight and the independent GPU chain
 # exist and are bound in a separately reviewed/frozen execution plan.
 pre=review('local_minimax_model32_preflight_result','actual_nogpu_preflight',p)
 q=read(I/'local_minimax_model32_preflight_result_review.json');rp=D/'run_preflight001';mon=D/'monitor_preflight001'
 for field,path in (('worker_sha256',rp/'worker_preflight.json'),('launch_sha256',rp/'launch.json'),('identity_sha256',rp/'container_identity.json'),('exit_sha256',rp/'exit.json'),('preflight_started_sha256',mon/'started.json'),('preflight_terminal_sha256',mon/'terminal.json')):
  if q.get(field)!=sha(path):raise ValueError('actual preflight proof binding '+field)
 if q.get('questions')!=32 or q.get('model_loaded') is not False or q.get('cuda_initialized') is not False or q.get('GPU_authorized') is not False:raise ValueError('actual preflight scope')
 formal=review('local_minimax_model32_formal_prelaunch','formal_prelaunch',p)
 fq=read(I/'local_minimax_model32_formal_prelaunch_review.json')
 if fq.get('verification_plan_sha256')!=sha(D/'verification_plan001.json'):raise ValueError('formal plan proof')
 gpu=review('local_minimax_model32_gpu_prelaunch','gpu_prelaunch',p)
 gq=read(I/'local_minimax_model32_gpu_prelaunch_review.json')
 if gq.get('preflight_review_sha256')!=pre['proof_sha256'] or gq.get('formal_review_sha256')!=formal['proof_sha256']:raise ValueError('GPU upstream proof')
 return {'scope':'gpu_run','plan_sha256':sha(D/'execution_plan001.json'),'review':prior,'preflight':pre,'formal':formal,'gpu':gpu}
