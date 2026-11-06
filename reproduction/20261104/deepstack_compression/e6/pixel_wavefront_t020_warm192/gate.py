"""Fail-closed CPU preflight and future GPU gates; neither queries a GPU."""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if len(c)<2 or (cwd/c[1]).resolve()!=script.resolve() or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or (out is not None and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervisor')
def source_check(source):
 if not isinstance(source,dict) or not source or any(sha(k)!=v for k,v in source.items()):raise ValueError('source mismatch')
def admit_cpu():
 plan=read(D/'preflight_plan001.json');proof=I/'pixel_t020_warm192_cpu_preflight_review.json';reviewer=I/'review_pixel_t020_warm192_cpu_preflight.py';monitor=I/'monitor_pixel_t020_warm192_cpu_preflight';q=read(proof);terminal(monitor,reviewer,proof);source_check(plan['source_sha256'])
 if plan.get('scope')!='warm192_noGPU_processor_preflight' or plan.get('GPU_authorized') is not False or q.get('status')!='passed' or q.get('independent_review') is not True or q.get('preflight_only') is not True or q.get('GPU_authorized') is not False or q.get('GT_opened') is not False or q.get('existing_DEV_predictions_opened') is not True or q.get('errors')!=[] or q.get('source_sha256')!=plan['source_sha256'] or q.get('plan_sha256')!=sha(D/'preflight_plan001.json') or q.get('reviewer_sha256')!=sha(reviewer):raise ValueError('CPU preflight independent review')
 return {'plan_sha256':sha(D/'preflight_plan001.json'),'review_sha256':sha(proof)}
def admit_gpu():
 planpath=D/'gpu_plan001.json';formal=D/'formal_plan001.json';analysis=D/'analysis_plan001.json';qpath=I/'pixel_t020_warm192_gpu_prelaunch_review.json';pre=I/'pixel_t020_warm192_preflight_result_review.json';arev=I/'pixel_t020_warm192_analysis_prelaunch_review.json'
 if not all(p.is_file() for p in (planpath,formal,analysis,qpath,pre,arev)):raise RuntimeError('GPU locked: independent preflight, formal, analysis and GPU review required')
 p=read(planpath);f=read(formal);a=read(analysis);q=read(qpath);r=read(pre);ar=read(arev);reviewer=I/'review_pixel_t020_warm192_gpu_prelaunch.py';monitor=I/'monitor_pixel_t020_warm192_gpu_prelaunch';pr=I/'review_pixel_t020_warm192_preflight_result.py';pm=I/'monitor_pixel_t020_warm192_preflight_result';air=I/'review_pixel_t020_warm192_analysis_prelaunch.py';aim=I/'monitor_pixel_t020_warm192_analysis_prelaunch'
 terminal(monitor,reviewer,qpath);terminal(pm,pr,pre);terminal(aim,air,arev);terminal(D/'monitor_preflight001',D/'preflight_launch.py',D/'run_preflight001')
 pre_fields={'plan_sha256':D/'preflight_plan001.json','worker_preflight_sha256':D/'run_preflight001/worker_preflight.json','launch_sha256':D/'run_preflight001/launch.json','identity_sha256':D/'run_preflight001/container_identity.json','exit_sha256':D/'run_preflight001/exit.json','started_sha256':D/'monitor_preflight001/started.json','terminal_sha256':D/'monitor_preflight001/terminal.json','reviewer_sha256':pr}
 if r.get('status')!='passed' or r.get('independent_review') is not True or r.get('preflight_only') is not True or r.get('GPU_authorized') is not False or r.get('GT_opened') is not False or r.get('existing_DEV_predictions_opened') is not True or r.get('errors')!=[] or any(r.get(k)!=sha(v) for k,v in pre_fields.items()):raise ValueError('noGPU preflight result review')
 if ar.get('status')!='passed' or ar.get('independent_review') is not True or ar.get('GT_opened') is not False or ar.get('timing_results_opened') is not False or ar.get('errors')!=[] or ar.get('plan_sha256')!=sha(analysis) or ar.get('reviewer_sha256')!=sha(air):raise ValueError('analysis review')
 source=p['source_sha256'];source_check(source);design=read(D/'design001.json');initial=read(B/'initial_plan.json')
 if p.get('scope')!='pixel_t020_warm192_eight_process_ttft' or p.get('jobs')!=design['processes'] or p.get('design_sha256')!=sha(D/'design001.json') or len(p['jobs'])!=8 or p.get('budget_end_unix')!=initial.get('budget_end_unix') or p.get('budget_source_sha256')!=sha(B/'initial_plan.json') or f.get('gpu_plan_sha256')!=sha(planpath) or a.get('gpu_plan_sha256')!=sha(planpath) or a.get('formal_plan_sha256')!=sha(formal):raise ValueError('frozen GPU/formal/analysis plan')
 fields={'plan_sha256':planpath,'formal_plan_sha256':formal,'analysis_plan_sha256':analysis,'analysis_review_sha256':arev,'preflight_result_sha256':pre,'reviewer_sha256':reviewer}
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('scope')!='pixel_t020_warm192_gpu_prelaunch' or q.get('GT_opened') is not False or q.get('scoring_performed') is not False or q.get('errors')!=[] or q.get('source_sha256')!=source or any(q.get(k)!=sha(v) for k,v in fields.items()):raise ValueError('GPU prelaunch review')
 weights=read(D/'run_preflight001/worker_preflight.json')['weights']
 if len(weights)!=4 or len({x['path'] for x in weights})!=4:raise ValueError('weight inventory')
 for item in weights:
  st=Path(item['path']).stat()
  if item['stat']!=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]:raise ValueError('weight endpoint changed')
 return {'plan_sha256':sha(planpath),'source_sha256':source,'jobs':p['jobs'],'plan':p}
if __name__=='__main__':raise SystemExit('GPU locked until independent reviews and complete plans')
