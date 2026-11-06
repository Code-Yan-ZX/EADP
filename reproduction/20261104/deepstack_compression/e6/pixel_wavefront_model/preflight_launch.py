"""Retained, supervised same-image no-GPU container preflight only."""
import argparse,importlib.util,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2];sys.path[:0]=[str(D),str(B/'e4_c_latency')]
import admission
if Path(admission.__file__).resolve()!=D/'admission.py':raise ValueError('foreign admission')
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if out!=(D/'run_preflight001').resolve():raise ValueError('unplanned preflight output')
 approval=admission.require_preflight();s=admission.read(D/'monitor_preflight001/started.json');cmd=s['command'];cwd=Path(s['cwd'])
 if s['worker_pid']!=os.getpid() or s['process_group']!=os.getpgrp() or os.getpid()!=os.getpgrp() or s['timeout_seconds']!=1500 or s['grace_seconds']!=90 or (cwd/cmd[1]).resolve()!=Path(__file__) or cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out:raise ValueError('preflight supervisor mismatch')
 timing=load('_wavefront_timing_launch',B/'e4_c_latency/timing_launch.py');initial=admission.read(B/'initial_plan.json');plan=admission.read(D/'preflight_plan001.json');budget={'budget_source':str(B/'initial_plan.json'),'budget_source_sha256':plan['budget_source_sha256'],'budget_start_unix':initial['budget_start_unix'],'budget_end_unix':initial['budget_end_unix']};timing.validate_budget_window(budget,time.time(),1200,1500,90)
 author=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';runtime=load('_wavefront_author_runtime',author.parent/'run_panel.py');out.mkdir(exist_ok=False);name='eadp-wavefront-model-cpu-'+str(os.getpid());argv=runtime.container_command(admission.read(author),[str(D/'preflight_entry.py'),'--out',str(out/'worker_preflight.json')],False,name)
 if argv.count('--rm')!=1 or '--gpus' in argv or '--device' in argv:raise ValueError('unexpected container devices/retention')
 argv.remove('--rm')
 with (out/'launch.json').open('x') as f:json.dump({'approval':approval,'controller':runtime.identity(os.getpid()),'command':argv,'name':name},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return timing.run_bounded(argv,name,1200,None,runtime.identity,out,require_gpu_idle=False)
if __name__=='__main__':raise SystemExit(main())
