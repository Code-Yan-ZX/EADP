"""Fail closed until the independently reviewed three-anchor GPU plan exists."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];P=B/'e6/rres_k256_pilot_v2';I=B/'e1_independent_review'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def supervisor(monitor,script,out):
 s=read(monitor/'started.json');t=read(monitor/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if s.get('worker_pid')!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve():raise ValueError('supervisor')
def admit_gpu():
 plan=read(D/'gpu_plan001.json')
 if plan.get('status')!='ready_for_independent_review' or plan.get('images')!=3 or plan.get('visual_tokens')!=256 or plan.get('GT_opened') is not False or plan.get('scoring_performed') is not False:raise ValueError('GPU plan')
 if any(sha(p)!=h for p,h in plan['source_sha256'].items()):raise ValueError('GPU plan source changed')
 prev=I/'rres_k256_processor_v2_preflight_result_review.json';reviewer=I/'review_rres_k256_processor_v2_preflight_result.py';monitor=I/'monitor_rres_k256_processor_v2_preflight_result';supervisor(monitor,reviewer,prev);proof=read(prev)
 if proof.get('status')!='passed' or proof.get('independent_review') is not True or proof.get('images')!=3 or proof.get('GT_opened') is not False or proof.get('model_loaded') is not False or proof.get('cuda_initialized') is not False or proof.get('GPU_authorized') is not False or proof.get('errors')!=[] or proof.get('worker_sha256')!=sha(P/'run_preflight001/worker_preflight.json') or proof.get('reviewer_sha256')!=sha(reviewer):raise ValueError('noGPU preflight not independently accepted')
 if plan.get('preflight_result_sha256')!=sha(prev) or plan.get('preflight_started_sha256')!=sha(monitor/'started.json') or plan.get('preflight_terminal_sha256')!=sha(monitor/'terminal.json'):raise ValueError('noGPU proof identity')
 prior=read(P/'run_preflight001/worker_preflight.json');stats=prior.get('weights_stats',{})
 if len(stats)!=4:raise ValueError('weight stat inventory')
 for name,expected in stats.items():
  s=Path(name).stat();actual=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
  if actual!=expected:raise ValueError('weight changed since validated preflight')
 runtime=I/'rres_k256_runtime_preflight_result_review.json';rs=I/'review_rres_k256_runtime_preflight_result.py';rm=I/'monitor_rres_k256_runtime_preflight_result';supervisor(rm,rs,runtime);rproof=read(runtime)
 if rproof.get('status')!='passed' or rproof.get('independent_review') is not True or rproof.get('GPU_authorized') is not False or rproof.get('GT_opened') is not False or rproof.get('errors')!=[] or rproof.get('worker_sha256')!=sha(B/'e6/rres_k256_runtime_preflight/run001/worker.json') or rproof.get('reviewer_sha256')!=sha(rs) or plan.get('runtime_preflight_result_sha256')!=sha(runtime) or plan.get('runtime_preflight_started_sha256')!=sha(rm/'started.json') or plan.get('runtime_preflight_terminal_sha256')!=sha(rm/'terminal.json'):raise ValueError('runtime noGPU proof')
 path=I/'rres_k256_gpu_prelaunch_review.json';script=I/'review_rres_k256_gpu_prelaunch.py';mon=I/'monitor_rres_k256_gpu_prelaunch';supervisor(mon,script,path);r=read(path)
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('images')!=3 or r.get('GT_opened') is not False or r.get('errors')!=[] or r.get('gpu_plan_sha256')!=sha(D/'gpu_plan001.json') or r.get('source_sha256')!=plan['source_sha256'] or r.get('reviewer_sha256')!=sha(script):raise ValueError('GPU independent review')
 return {'gpu_plan_sha256':sha(D/'gpu_plan001.json'),'preflight_result_sha256':sha(prev),'gpu_prelaunch_sha256':sha(path)}
