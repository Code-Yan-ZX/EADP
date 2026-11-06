"""Retained, supervised Docker controller for CPU preflight or GPU diagnostic."""
import argparse,hashlib,importlib.util,json,os,shutil,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=BASE.parents[2]
def check_gate(path):
 if hashlib.sha256(Path(path).read_bytes()).hexdigest()!='fdd588a9914cb624a4fb496158f4166a5b571a64db8a6dfacb8db316471920c6':raise ValueError('gate source changed before import')
 return path
spec=importlib.util.spec_from_file_location('_budget_launcher_gate',check_gate(HERE/'gate.py'));gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)

def preflight_supervisor(s):
 cmd=gate.resolved_command(s)
 if (s.get('worker_pid')!=os.getpid() or s.get('process_group')!=os.getpgrp() or os.getpid()!=os.getpgrp() or
     s.get('timeout_seconds')!=900 or s.get('grace_seconds')!=90 or s.get('stale_seconds') is not None or
     cmd.count(str(HERE/'launch.py'))!=1 or s['command'].count('--preflight-only')!=1):raise ValueError('CPU supervisor mismatch')

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--preflight-only',action='store_true');args=p.parse_args()
 if not args.run_id.isalnum():raise ValueError('invalid run id')
 plan=gate.plan();gate.prior();gate.execution_cpu(plan);approval=None
 if args.preflight_only:
  gate.admit_cpu_preflight(plan)
  if args.run_id!='001':raise ValueError('fixed CPU preflight id')
  out=HERE/'run_preflight001';monitor=HERE/'monitor_preflight001';preflight_supervisor(gate.read(monitor/'started.json'));inner=780
 else:
  approval=gate.admit_gpu();out=HERE/('run_'+args.run_id);monitor=HERE/('monitor_run'+args.run_id)
  gate.check_supervisor(gate.read(monitor/'started.json'),args.run_id,time.time(),plan);inner=plan['inner_timeout_seconds']
 if shutil.disk_usage(HERE).free<plan['minimum_free_bytes']:raise ValueError('disk space insufficient')
 bootpath=BASE/'e6/pixel_natural_timing_pilot_v4/bootstrap.py'
 if gate.sha(bootpath)!='ab81e2f48f995403afc21ab6ab4875c8b4ae4e9bcca72be063ab8fc42a53fdbc':raise ValueError('bootstrap source changed')
 spec=importlib.util.spec_from_file_location('_budget_execution_boot',bootpath);boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.pin_all()
 from timing_launch import run_bounded,validate_budget_window
 initial=gate.read(BASE/'initial_plan.json');budget={'budget_source':str(BASE/'initial_plan.json'),'budget_source_sha256':plan['budget_source_sha256'],
  'budget_start_unix':initial['budget_start_unix'],'budget_end_unix':initial['budget_end_unix']}
 validate_budget_window(budget,time.time(),inner,900 if args.preflight_only else plan['outer_timeout_seconds'],90 if args.preflight_only else plan['grace_seconds'])
 author=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';runtime_path=author.parent/'run_panel.py'
 if gate.sha(runtime_path)!='aeee2a27c4394fc683245dce6fe2629f2fa1d9948ab66bb68edbdd46a9c65645':raise ValueError('container runtime source changed')
 spec=importlib.util.spec_from_file_location('_budget_execution_runtime',runtime_path);runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
 if not args.preflight_only and runtime.gpu_pids()!=[]:raise ValueError('GPU occupied')
 out.mkdir(exist_ok=False);name='eadp-fixedref-'+('cpu-' if args.preflight_only else 'gpu-')+args.run_id+'-'+str(os.getpid())
 worker=HERE/'worker.py';command=[str(worker),'--out',str(out)]+(['--preflight-only'] if args.preflight_only else [])
 argv=runtime.container_command(gate.read(author),command,not args.preflight_only,name)
 if argv.count('--rm')!=1:raise ValueError('unexpected container retention')
 argv.remove('--rm')
 with (out/'launch.json').open('x') as f:
  json.dump({'stage':'CPU worker preflight' if args.preflight_only else '222-input fixed32 GPU reference','command':argv,'container_name':name,
             'controller':runtime.identity(os.getpid()),'run_out':str(out),'plan_sha256':gate.sha(HERE/'execution_plan001.json'),
             'worker_sha256':gate.sha(worker),'launcher_sha256':gate.sha(__file__),'approval_bundle':approval,'timeout_seconds':inner},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return run_bounded(argv,name,inner,None if args.preflight_only else runtime.gpu_pids,runtime.identity,out,require_gpu_idle=not args.preflight_only)
if __name__=='__main__':raise SystemExit(main())
