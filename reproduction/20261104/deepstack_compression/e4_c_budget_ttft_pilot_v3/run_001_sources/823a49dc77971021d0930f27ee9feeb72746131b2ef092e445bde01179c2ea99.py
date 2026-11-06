"""Nine bounded, sequential, exclusive GPU containers; proof-gated prelaunch."""
import importlib.util,json,os,shutil,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];ROOT=BASE.parents[2]
spec=importlib.util.spec_from_file_location('_pilot_bootstrap_launch_v4',HERE/'bootstrap.py');boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.pin_all()
pre=boot.pin('preflight',HERE/'preflight.py');worker=boot.pin('worker',HERE/'worker.py');adm=boot.pin('pilot_admission_v4',HERE/'admission.py')
from timing_launch import validate_budget_window
gpu_bounded=boot.pin('gpu_bounded_v4',HERE/'gpu_bounded.py')
run_bounded=gpu_bounded.run_bounded
INNER=1200;OUTER=11700;GRACE=120

def validate_supervisor(start,pid,pgid,cwd):
 command=start.get('command',[]);scwd=Path(start.get('cwd','/'));resolved=[str((scwd/x).resolve()) if not Path(x).is_absolute() else str(Path(x).resolve()) for x in command]
 if (start.get('worker_pid')!=pid or start.get('process_group')!=pgid or pid!=pgid or
     start.get('timeout_seconds')!=OUTER or start.get('grace_seconds')!=GRACE or start.get('snapshot_seconds')!=3600 or start.get('poll_seconds')!=5 or
     resolved.count(str(HERE/'gpu_launch.py'))!=1 or Path(cwd).resolve()!=scwd.resolve()):raise ValueError('timing supervisor mismatch')
 return True

def check_stats(proof):
 for item in proof['weight_stats']:
  st=Path(item['path']).stat();now=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
  if now!=item['stat']:raise ValueError('weight stat changed since preflight')

def validate_worker_journal(path,job):
 rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
 expected=['model_loading','model_ready','started','raw_generation','completed','started','raw_generation','completed','job_complete']
 if [r.get('event') for r in rows]!=expected or any(r.get('job')!=job for r in rows):raise ValueError('worker lifecycle incomplete')
 for start in (2,5):
  phase='warmup' if start==2 else 'measured';segment=rows[start:start+3]
  if any(x.get('phase')!=phase for x in segment):raise ValueError('worker phase identity')
  value=segment[2].get('result',{});k=value.get('dynamic_k');observed=value.get('count_observer_enabled')
  if (value.get('processor_verified') is not True or not isinstance(value.get('tokens'),list) or not value['tokens'] or
      not (value.get('eos') is True or value.get('truncated') is True) or
      (phase=='warmup' and (observed is not False or k is not None)) or
      (phase=='measured' and (observed is not True or type(k) is not int or not 1<=k<=1024))):raise ValueError('worker result incomplete')
 return True

def release_worker(job,jobdir,launch,identity,controller_identity,supervisor,supervisor_validator=None):
 (supervisor_validator or validate_supervisor)(supervisor,os.getpid(),os.getpgrp(),os.getcwd())
 host=identity.get('host_init') or {};cid=identity.get('container_id');argv=identity.get('host_argv',[])
 if (launch.get('controller')!=controller_identity or not isinstance(cid,str) or len(cid)!=64 or
     cid not in identity.get('host_cgroup','') or type(host.get('pid')) is not int or host['pid']<=0 or
     type(host.get('start_ticks')) is not int or host['start_ticks']<=0 or not host.get('boot_id') or
     argv.count(str(HERE/'gpu_worker_candidate.py'))!=1 or argv.count('--job-index')!=1 or argv[argv.index('--job-index')+1]!=str(job['order'])):raise ValueError('container/controller handshake identity')
 release={'plan_sha256':launch['plan_sha256'],'launch_sha256':pre.sha(jobdir/'launch.json'),'job':job,'controller':controller_identity,'container_id':cid,'host_init':host}
 temp=jobdir/'release.pending';final=jobdir/'release.json'
 with temp.open('x') as f:json.dump(release,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.link(temp,final);temp.unlink()
 return release

def run_jobs(plan,runtime,out,budget_end,journal):
 identities=set()
 for job in plan['jobs']:
  now=time.time();remaining=len(plan['jobs'])-job['order']
  if now+remaining*INNER+GRACE>budget_end:raise ValueError('remaining user budget insufficient')
  if runtime.gpu_pids()!=[]:raise ValueError('GPU busy before job')
  jobdir=out/('job_%03d'%job['order']);jobdir.mkdir(exist_ok=False)
  name='eadp-pixel-ttft-%03d-%d'%(job['order'],os.getpid())
  author=pre.read(ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json')
  argv=runtime.container_command(author,[str(HERE/'gpu_worker_candidate.py'),'--job-index',str(job['order'])],True,name)
  if argv.count('--rm')!=1:raise ValueError('unexpected Docker retention')
  argv.remove('--rm')
  record={'plan_sha256':pre.sha(HERE/'plan001.json'),'job':job,'command':argv,'container_name':name,'controller':runtime.identity(os.getpid())}
  with (jobdir/'launch.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  worker.append(journal,{'event':'job_started','job':job})
  def release(identity):
   release_worker(job,jobdir,record,identity,runtime.identity(os.getpid()),pre.read(HERE/'monitor_pilot001/started.json'))
  code=run_bounded(argv,name,INNER,runtime.gpu_pids,runtime.identity,jobdir,require_gpu_idle=True,on_identity=release)
  result=pre.read(jobdir/'exit.json');host=(result.get('container_identity') or {}).get('host_init') or {};identity=(host.get('pid'),host.get('start_ticks'),host.get('boot_id'))
  if (code!=0 or result.get('returncode')!=0 or result.get('container_exit_code')!=0 or result.get('container_stopped_verified') is not True or result.get('gpu_pids_after')!=[] or result.get('timed_out') is not False or result.get('interrupted') is not None or result.get('client_exited_while_container_running') is not False or result.get('failure_type') is not None or type(identity[0]) is not int or identity[0]<=0 or type(identity[1]) is not int or identity[1]<=0 or not identity[2] or identity in identities):
   worker.append(journal,{'event':'job_failed','job':job,'launcher_code':code,'exit':result});raise ValueError('GPU job did not exit cleanly or reused process identity')
  validate_worker_journal(jobdir/'observations/journal.jsonl',job)
  identities.add(identity);worker.append(journal,{'event':'job_exited','job':job,'host_init':host})
 if len(identities)!=9:raise ValueError('incomplete independent workers')
 return len(identities)

def main():
 # Independent evidence and budget must precede GPU query and output creation.
 admission=adm.require();plan=pre.read(HERE/'plan001.json')
 start=pre.read(HERE/'monitor_pilot001/started.json');validate_supervisor(start,os.getpid(),os.getpgrp(),os.getcwd())
 initial=pre.read(BASE/'initial_plan.json');now=time.time();budget={'budget_source':str(BASE/'initial_plan.json'),'budget_source_sha256':plan['budget_source_sha256'],'budget_start_unix':initial['budget_start_unix'],'budget_end_unix':initial['budget_end_unix']}
 validate_budget_window(budget,now,OUTER-GRACE,OUTER,GRACE)
 if now+OUTER+GRACE>initial['budget_end_unix']:raise ValueError('budget lacks supervisor cleanup grace')
 proof=pre.read(HERE/'preflight001.json');check_stats(proof)
 if shutil.disk_usage(HERE).free<32*1024**3:raise ValueError('insufficient disk')
 author=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';spec=importlib.util.spec_from_file_location('_pixel_pilot_runtime',author.parent/'run_panel.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
 if runtime.gpu_pids()!=[]:raise ValueError('GPU busy')
 out=HERE/'run_pilot001';out.mkdir(exist_ok=False);journal=out/'controller.jsonl'
 worker.append(journal,{'event':'controller_started','admission':admission,'controller':runtime.identity(os.getpid())})
 try:
  count=run_jobs(plan,runtime,out,initial['budget_end_unix'],journal)
  check_stats(proof);pre.verify_source_plan(plan)
  worker.append(journal,{'event':'controller_complete','jobs':count})
 except BaseException as exc:
  import traceback
  worker.append(journal,{'event':'controller_failed','error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()});raise
 return 0
if __name__=='__main__':raise SystemExit(main())
