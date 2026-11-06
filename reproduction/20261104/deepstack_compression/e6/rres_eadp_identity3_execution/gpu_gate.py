"""Future GPU gate. No plan or independent approval exists yet."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];IDENTITY=ROOT/'e6/rres_eadp_identity3_v2';REV=ROOT/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ended(s,t):
 a=read(s);b=read(t);return type(a.get('worker_pid')) is int and b.get('worker_pid')==a['worker_pid'] and b.get('reason')=='completed' and b.get('worker_returncode')==0 and b.get('remaining_process_group')==[]
def admit():
 p=read(HERE/'gpu_plan001.json')
 if p.get('scope')!='rres_eadp_identity3_gpu' or any(sha(path)!=digest for path,digest in p['source_sha256'].items()):raise ValueError('GPU plan source')
 prior=REV/'rres_eadp_identity3_v2_preflight_result_review.json';q=read(prior);rmon=REV/'monitor_rres_eadp_identity3_v2_preflight_result';run=IDENTITY/'preflight001';mon=IDENTITY/'monitor_preflight001'
 files={'plan_sha256':IDENTITY/'preflight_plan001.json','worker_sha256':run/'worker_preflight.json','launch_sha256':run/'launch.json','identity_sha256':run/'container_identity.json','exit_sha256':run/'exit.json','started_sha256':mon/'started.json','terminal_sha256':mon/'terminal.json','prelaunch_review_sha256':REV/'rres_eadp_identity3_v2_cpu_preflight_review.json','reviewer_sha256':REV/'review_rres_eadp_identity3_v2_preflight_result.py'}
 w=read(run/'worker_preflight.json');e=read(run/'exit.json')
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('preflight_only') is not True or q.get('GPU_authorized') is not False or q.get('GT_opened') is not False or q.get('existing_predictions_opened') is not True or q.get('errors')!=[] or any(q.get(k)!=sha(v) for k,v in files.items()) or not ended(mon/'started.json',mon/'terminal.json') or not ended(rmon/'started.json',rmon/'terminal.json') or p['preflight_review_sha256']!=sha(prior) or w.get('status')!='passed' or w.get('model_loaded') is not False or w.get('cuda_initialized') is not False or e.get('container_exit_code')!=0 or e.get('container_stopped_verified') is not True:raise ValueError('noGPU identity evidence')
 proof=REV/'rres_eadp_identity3_gpu_prelaunch_review.json';r=read(proof);reviewer=REV/'review_rres_eadp_identity3_gpu_prelaunch.py';rmon=REV/'monitor_rres_eadp_identity3_gpu_prelaunch'
 formal=HERE/'verification_plan001.json';f=read(formal)
 if f.get('gpu_plan_sha256')!=sha(HERE/'gpu_plan001.json') or f.get('verifier_sha256')!=sha(HERE/'verify.py') or f.get('snapshot_sha256')!=sha(HERE/'snapshot.py'):raise ValueError('formal plan')
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('scope')!='identity3_gpu_prelaunch' or r.get('errors')!=[] or r.get('GT_opened') is not False or r.get('plan_sha256')!=sha(HERE/'gpu_plan001.json') or r.get('verification_plan_sha256')!=sha(formal) or r.get('reviewer_sha256')!=sha(reviewer) or r.get('source_sha256')!=p['source_sha256'] or not ended(rmon/'started.json',rmon/'terminal.json'):raise ValueError('independent GPU prelaunch')
 return {'scope':p['scope'],'plan_sha256':sha(HERE/'gpu_plan001.json'),'review_sha256':sha(proof),'preflight_review_sha256':sha(prior)}
