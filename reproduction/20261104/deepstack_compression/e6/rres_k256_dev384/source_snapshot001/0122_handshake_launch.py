"""Real retained Docker + production release handshake, without GPU device."""
import importlib.util,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];ROOT=BASE.parents[2]
spec=importlib.util.spec_from_file_location('_pixel_handshake_bootstrap_v4',HERE/'bootstrap.py');boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.pin_all()
pre=boot.pin('preflight',HERE/'preflight.py');launch=boot.pin('gpu_launch_v4',HERE/'gpu_launch.py')

def validate_supervisor(s,pid,pgid,cwd):
 cmd=s.get('command',[]);root=Path(s.get('cwd','/'));resolved=[str((root/x).resolve()) if not Path(x).is_absolute() else str(Path(x).resolve()) for x in cmd]
 if (s.get('worker_pid')!=pid or s.get('process_group')!=pgid or pid!=pgid or s.get('timeout_seconds')!=300 or s.get('grace_seconds')!=90 or resolved.count(str(HERE/'handshake_launch.py'))!=1 or root.resolve()!=Path(cwd).resolve()):raise ValueError('handshake supervisor mismatch')
 return True

def main():
 plan=pre.read(HERE/'plan001.json');pre.verify_source_plan(plan);job=plan['jobs'][0]
 mon=HERE/'monitor_handshake001';start=pre.read(mon/'started.json');validate_supervisor(start,os.getpid(),os.getpgrp(),os.getcwd())
 author=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';spec=importlib.util.spec_from_file_location('_pixel_handshake_runtime',author.parent/'run_panel.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
 root=HERE/'run_handshake001';root.mkdir(exist_ok=False);jobdir=root/'job_000';jobdir.mkdir(exist_ok=False);name='eadp-pixel-handshake-'+str(os.getpid())
 argv=runtime.container_command(pre.read(author),[str(HERE/'gpu_worker_candidate.py'),'--job-index','0','--handshake-only'],False,name)
 if argv.count('--rm')!=1:raise ValueError('unexpected container retention')
 argv.remove('--rm');record={'plan_sha256':pre.sha(HERE/'plan001.json'),'job':job,'command':argv,'controller':runtime.identity(os.getpid())}
 with (jobdir/'launch.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 def release(identity):launch.release_worker(job,jobdir,record,identity,runtime.identity(os.getpid()),pre.read(mon/'started.json'),supervisor_validator=validate_supervisor)
 return launch.run_bounded(argv,name,180,None,runtime.identity,jobdir,require_gpu_idle=False,on_identity=release)
if __name__=='__main__':raise SystemExit(main())
