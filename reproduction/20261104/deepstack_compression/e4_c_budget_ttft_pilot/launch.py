"""Bounded retained Docker with timed health windows, no implicit GPU access."""
import argparse,importlib.util,json,os,shutil,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=BASE.parents[2]
spec=importlib.util.spec_from_file_location('_budget_ttft_launch_admission',HERE/'admission.py');admission=importlib.util.module_from_spec(spec);spec.loader.exec_module(admission)
def supervisor(s,runid,is_cpu,parent_argv,pid,pgid):
 cmd=s['command'];cwd=Path(s['cwd']);resolved=[str((cwd/x).resolve()) if not Path(x).is_absolute() else str(Path(x).resolve()) for x in cmd]
 parent_resolved=[str((cwd/x).resolve()) if not Path(x).is_absolute() else str(Path(x).resolve()) for x in parent_argv]
 if (s.get('worker_pid')!=pid or s.get('process_group')!=pgid or pid!=pgid or resolved.count(str(HERE/'launch.py'))!=1 or
     cmd.count('--run-id')!=1 or cmd[cmd.index('--run-id')+1]!=runid or
     parent_resolved.count(str(BASE/'e4_c_latency/timing_supervise.py'))!=1 or s.get('snapshot_seconds')!=3600 or s.get('poll_seconds')!=5 or s.get('grace_seconds')!=90):raise ValueError('timed supervisor identity mismatch')
 if is_cpu:
  if s.get('timeout_seconds')!=900 or s.get('stale_seconds') is not None or cmd.count('--preflight-only')!=1:raise ValueError('CPU supervisor mismatch')
 else:
  if s.get('timeout_seconds')!=3720 or s.get('stale_seconds')!=1200 or s.get('progress_paths')!=[str(HERE/('run_'+runid)/'journal.jsonl')] or '--preflight-only' in cmd:raise ValueError('GPU supervisor mismatch')
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--preflight-only',action='store_true');args=p.parse_args()
 if not args.run_id.isalnum():raise ValueError('invalid run id')
 plan=admission.plan();approval=None
 if args.preflight_only:
  admission.identity.admit_identity()
  if args.run_id!='001':raise ValueError('fixed CPU preflight ID')
  out=HERE/'run_preflight001';monitor=HERE/'monitor_preflight001';inner,outer=780,900
 else:
  approval=admission.admit_gpu();out=HERE/('run_'+args.run_id);monitor=HERE/('monitor_run'+args.run_id);inner,outer=3600,3720
 parent=[x.decode(errors='replace') for x in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().split(b'\0') if x]
 supervisor(admission.read(monitor/'started.json'),args.run_id,args.preflight_only,parent,os.getpid(),os.getpgrp())
 if shutil.disk_usage(HERE).free<plan['minimum_free_bytes']:raise ValueError('insufficient disk space')
 bootpath=BASE/'e6/pixel_natural_timing_pilot_v4/bootstrap.py';spec=importlib.util.spec_from_file_location('_budget_ttft_launch_boot',bootpath);boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.pin_all()
 from timing_launch import run_bounded,validate_budget_window
 initial=admission.read(BASE/'initial_plan.json');validate_budget_window({'budget_source':str(BASE/'initial_plan.json'),'budget_source_sha256':plan['budget_source_sha256'],
  'budget_start_unix':initial['budget_start_unix'],'budget_end_unix':initial['budget_end_unix']},time.time(),inner,outer,90)
 author=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';spec=importlib.util.spec_from_file_location('_budget_ttft_docker',author.parent/'run_panel.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
 if not args.preflight_only and runtime.gpu_pids()!=[]:raise ValueError('GPU occupied')
 out.mkdir(exist_ok=False);name='eadp-budget-ttft-'+('cpu-' if args.preflight_only else 'gpu-')+args.run_id+'-'+str(os.getpid());worker=HERE/'worker.py'
 cmd=[str(worker),'--out',str(out)]+(['--preflight-only'] if args.preflight_only else [])
 argv=runtime.container_command(admission.read(author),cmd,not args.preflight_only,name)
 if argv.count('--rm')!=1:raise ValueError('retention mismatch')
 argv.remove('--rm')
 with (out/'launch.json').open('x') as f:
  json.dump({'stage':'CPU preflight' if args.preflight_only else '3-anchor TTFT','command':argv,'container_name':name,'controller':runtime.identity(os.getpid()),
             'run_out':str(out),'approval_bundle':approval,'worker_sha256':admission.identity.sha(worker),'launcher_sha256':admission.identity.sha(__file__),
             'runtime_plan_sha256':admission.identity.sha(HERE/'runtime_plan001.json'),'time_namespace':os.readlink('/proc/self/ns/time'),'timeout_seconds':inner},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return run_bounded(argv,name,inner,None if args.preflight_only else runtime.gpu_pids,runtime.identity,out,require_gpu_idle=not args.preflight_only)
if __name__=='__main__':raise SystemExit(main())
