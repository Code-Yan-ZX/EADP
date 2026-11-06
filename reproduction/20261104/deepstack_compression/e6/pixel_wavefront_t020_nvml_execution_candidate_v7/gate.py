"""Prospective GPU gate: no device query, fails closed without future proofs."""
import hashlib,importlib.util,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review';P=B/'e6/pixel_wavefront_t020_nvml_execution_preflight';A=B/'e6/pixel_wavefront_t020_nvml_execution_accept'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if len(cmd)<2 or (cwd/cmd[1]).resolve()!=script.resolve() or type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervisor evidence')
def admit_gpu():
 planpath=D/'gpu_plan001.json';reviewpath=I/'pixel_t020_nvml_gpu_prelaunch_review.json';reviewer=I/'review_pixel_t020_nvml_gpu_prelaunch.py';monitor=I/'monitor_pixel_t020_nvml_gpu_prelaunch';formal=A/'verification_plan001.json';pre=I/'pixel_t020_nvml_execution_preflight_result_review.json'
 if not all(p.is_file() for p in (planpath,reviewpath,formal,pre)):raise RuntimeError('GPU locked: missing frozen plan, formal or independent proof')
 p=read(planpath);q=read(reviewpath);f=read(formal);pr=read(pre);terminal(monitor,reviewer,reviewpath)
 pre_reviewer=I/'review_pixel_t020_nvml_execution_preflight_result.py';pre_mon=I/'monitor_pixel_t020_nvml_execution_preflight_result';terminal(pre_mon,pre_reviewer,pre);terminal(P/'monitor_preflight001',P/'launch.py',P/'run_preflight001')
 pre_fields={'plan_sha256':P/'plan001.json','worker_preflight_sha256':P/'run_preflight001/worker_preflight.json','launch_sha256':P/'run_preflight001/launch.json','identity_sha256':P/'run_preflight001/container_identity.json','exit_sha256':P/'run_preflight001/exit.json','started_sha256':P/'monitor_preflight001/started.json','terminal_sha256':P/'monitor_preflight001/terminal.json','reviewer_sha256':pre_reviewer}
 if pr.get('status')!='passed' or pr.get('independent_review') is not True or pr.get('preflight_only') is not True or pr.get('GPU_authorized') is not False or pr.get('GT_opened') is not False or pr.get('existing_DEV_predictions_opened') is not True or pr.get('errors')!=[] or any(pr.get(k)!=sha(v) for k,v in pre_fields.items()):raise ValueError('noGPU independent proof')
 source=p.get('source_sha256');design=B/'e6/pixel_wavefront_t020_nvml_nine/design001.json';jobs=read(design)['jobs']
 if p.get('scope')!='pixel_t020_t0_nine_process_nvml' or p.get('jobs')!=jobs or len(jobs)!=9 or p.get('design_sha256')!=sha(design) or not isinstance(source,dict) or not source or any(sha(k)!=v for k,v in source.items()):raise ValueError('GPU source or exact job schedule')
 initial=read(B/'initial_plan.json')
 if p.get('budget_end_unix')!=initial.get('budget_end_unix') or p.get('budget_source_sha256')!=sha(B/'initial_plan.json') or p.get('job_timeout_seconds')!=2400 or p.get('job_grace_seconds')!=240 or type(p.get('outer_timeout_seconds')) is not int or p['outer_timeout_seconds']<28800 or type(p.get('grace_seconds')) is not int or p['grace_seconds']<600:raise ValueError('budget or recovery limits')
 if f.get('gpu_plan_sha256')!=sha(planpath) or f.get('GPU_used') is not False or f.get('GT_opened') is not False:raise ValueError('formal plan')
 fields={'plan_sha256':planpath,'reviewer_sha256':reviewer,'formal_plan_sha256':formal,'preflight_result_sha256':pre}
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('scope')!='pixel_nine_process_nvml_gpu_prelaunch' or q.get('errors')!=[] or q.get('GT_opened') is not False or q.get('scoring_performed') is not False or q.get('existing_DEV_predictions_opened') is not True or q.get('source_sha256')!=source or any(q.get(k)!=sha(v) for k,v in fields.items()):raise ValueError('GPU review')
 prior=(B/'e6/pixel_wavefront_t020_timing_final_v9/gpu_gate.py').resolve();name='_pixel_nvml_prior_weight_gate'
 if name in sys.modules or sha(prior)!=source[str(prior)]:raise ValueError('weight gate source')
 sp=importlib.util.spec_from_file_location(name,prior);m=importlib.util.module_from_spec(sp);sys.modules[name]=m
 try:sp.loader.exec_module(m);w=m.weight_preflight()
 finally:del sys.modules[name]
 if p.get('weight_reference_sha256')!=w['worker_preflight_sha256'] or p.get('weight_files')!=[x['path'] for x in w['expected']]:raise ValueError('four weight endpoint stats')
 return {'plan_sha256':sha(planpath),'plan':p,'jobs':jobs,'source_sha256':source}
if __name__=='__main__':raise SystemExit('no GPU authority without reviewed prelaunch')
