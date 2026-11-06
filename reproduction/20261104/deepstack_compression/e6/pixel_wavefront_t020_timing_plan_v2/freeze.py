"""Freeze future GPU plan after noGPU result; never query GPU."""
import hashlib,importlib.util,json,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];R=B/'e6/pixel_wavefront_t020_timing_final_v7';A=B/'e6/pixel_wavefront_t020_timing_accept_v5';I=B/'e1_independent_review';C=B/'e6/pixel_wavefront_t020_model_timing_v2';PY='/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def validate_schedule(jobs):
 if len(jobs)!=24 or [j['order'] for j in jobs]!=list(range(24)) or sum(j['phase']=='warmup' for j in jobs)!=6 or sum(j['phase']=='measurement' for j in jobs)!=18 or len({(j['task'],str(j['qid']),j['arm'],j['phase'],j['repeat']) for j in jobs})!=24:raise ValueError('schedule')
 return True
def freeze():
 out=R/'gpu_plan001.json';result=I/'pixel_wavefront_t020_timing_final_v7_preflight_result_review.json';reviewer=I/'review_pixel_wavefront_t020_timing_final_v7_preflight_result.py';mon=I/'monitor_pixel_wavefront_t020_timing_final_v7_preflight_result'
 if out.exists() or not result.exists():raise ValueError('existing plan or noGPU review absent')
 q=read(result);pre=read(R/'preflight_plan001.json');s=read(mon/'started.json');t=read(mon/'terminal.json');cmd=s['command'];cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (cwd/cmd[1]).resolve()!=reviewer.resolve() or cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=result.resolve():raise ValueError('noGPU review supervisor')
 fields={'plan_sha256':R/'preflight_plan001.json','worker_preflight_sha256':R/'run_preflight001/worker_preflight.json','launch_sha256':R/'run_preflight001/launch.json','identity_sha256':R/'run_preflight001/container_identity.json','exit_sha256':R/'run_preflight001/exit.json','started_sha256':R/'monitor_preflight001/started.json','terminal_sha256':R/'monitor_preflight001/terminal.json','prelaunch_review_sha256':I/'pixel_wavefront_t020_timing_final_v7_preflight_review.json','reviewer_sha256':reviewer}
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('preflight_only') is not True or q.get('gpu_authorized') is not False or q.get('GT_opened') is not False or q.get('existing_DEV_predictions_opened') is not True or q.get('errors')!=[] or q.get('source_sha256')!=pre['source_sha256'] or any(q.get(k)!=sha(v) for k,v in fields.items()):raise ValueError('noGPU review')
 actual={p:sha(p) for p in pre['source_sha256']}
 if actual!=pre['source_sha256']:raise ValueError('source changed')
 gp=R/'gpu_gate.py'
 if actual.get(str(gp.resolve()))!=sha(gp):raise ValueError('gate source')
 sp=importlib.util.spec_from_file_location('_pixel_t020_plan_gpu_gate',gp);gate=importlib.util.module_from_spec(sp);sys.modules[sp.name]=gate
 try:sp.loader.exec_module(gate);w=gate.weight_preflight()
 except BaseException:del sys.modules[sp.name];raise
 own=read(D/'cpu_proof001.json'); own_start=read(D/'monitor_cpu001/started.json'); own_term=read(D/'monitor_cpu001/terminal.json')
 if own.get('status')!='passed' or own['source_sha256_before']!=own['source_sha256_after'] or any(sha(k)!=v for k,v in own['source_sha256_after'].items()) or own_term.get('reason')!='completed' or own_term.get('worker_returncode')!=0 or own_term.get('remaining_process_group')!=[] or own_start.get('worker_pid')!=own_term.get('worker_pid') or (Path(own_start['cwd'])/own_start['command'][1]).resolve()!=D/'cpu_check.py':raise ValueError('plan CPU proof')
 cpu=read(A/'cpu_proof001.json');s=read(A/'monitor_cpu001/started.json');t=read(A/'monitor_cpu001/terminal.json')
 if cpu.get('status')!='passed' or cpu.get('GT_opened') is not False or cpu.get('GPU_used') is not False or cpu.get('predictions_opened') is not False or cpu['source_sha256_before']!=cpu['source_sha256_after'] or any(sha(k)!=v for k,v in cpu['source_sha256_after'].items()) or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (Path(s['cwd'])/s['command'][1]).resolve()!=A/'cpu_check.py':raise ValueError('formal CPU proof')
 extras=[D/'freeze.py',D/'test_freeze.py',D/'cpu_check.py',D/'cpu_proof001.json',D/'monitor_cpu001/started.json',D/'monitor_cpu001/terminal.json',B/'e4_c_latency/timing_supervise.py',B/'monitor/supervise.py',B/'e6/pixel_natural_timing_pilot_v4/gpu_bounded.py',B/'e6/pixel_wavefront_t020_timing_controller_draft/job_contract.py',B/'initial_plan.json',A/'cpu_proof001.json',A/'monitor_cpu001/started.json',A/'monitor_cpu001/terminal.json',result,reviewer,mon/'started.json',mon/'terminal.json',I/'pixel_wavefront_model_t020_preflight_result_review.json',I/'review_pixel_wavefront_model_t020_preflight_result.py',I/'monitor_pixel_wavefront_model_t020_preflight_result/started.json',I/'monitor_pixel_wavefront_model_t020_preflight_result/terminal.json',B/'e6/pixel_wavefront_model_t020/run_preflight001/worker_preflight.json']
 for p in extras:actual[str(p.resolve())]=sha(p)
 actual.update(cpu['source_sha256_after'])
 if any(sha(k)!=v for k,v in actual.items()):raise ValueError('source race')
 jobs=read(C/'design001.json')['jobs'];validate_schedule(jobs);cmd=[PY,str(R/'launch.py'),'--run-id','001']
 plan={'scope':'pixel_t020_isolated_process_ttft_24','jobs':jobs,'design_sha256':sha(C/'design001.json'),'source_sha256':actual,'weight_reference_sha256':w['worker_preflight_sha256'],'weight_files':[x['path'] for x in w['expected']],'budget_source_sha256':sha(B/'initial_plan.json'),'run_supervisor_command':cmd,'run_supervisor_cwd':str(B),'inner_timeout_seconds':1200,'outer_timeout_seconds':29400,'grace_seconds':120,'cleanup_seconds':120,'snapshot_seconds':3600,'poll_seconds':5,'minimum_disk_bytes':32*1024**3,'GPU_authorized':False,'GT_opened':False,'predictions_opened':True,'limitations':['separate process warmup is not in-process steady state','three anchors and three repeats post-hoc exploratory','selector clone included in both arms','unobserved activity unknown','stat endpoints cannot exclude transient weight replacement']}
 with out.open('x') as f:json.dump(plan,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':freeze()
