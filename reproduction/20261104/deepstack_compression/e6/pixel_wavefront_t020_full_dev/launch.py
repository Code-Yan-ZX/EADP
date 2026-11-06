"""Bounded retained Docker launch; GPU stays closed until both independent reviews."""
import argparse
import importlib.util
import os
import shutil
import sys
import time
from pathlib import Path
import admission

D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2]
AUTHOR_PLAN=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
sys.path.insert(0,str(B/'e4_c_latency'))
from timing_launch import run_bounded, validate_budget_window

def runtime_module():
    path=AUTHOR_PLAN.parent/'run_panel.py';diagnose=(AUTHOR_PLAN.parent/'diagnose.py').resolve()
    previous=sys.modules.get('diagnose')
    if previous is not None and Path(getattr(previous,'__file__','/')).resolve()!=diagnose:
        raise ValueError('foreign cached diagnose module')
    spec=importlib.util.spec_from_file_location('_wavefront_panel_runtime',path)
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    if Path(sys.modules['diagnose'].__file__).resolve()!=diagnose:raise ValueError('actual diagnose path mismatch')
    return runtime

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['preflight','run']);parser.add_argument('--dependency-smoke',action='store_true');a=parser.parse_args()
    if bool(a.stage)==bool(a.dependency_smoke):raise ValueError('choose stage or dependency smoke')
    p=admission.check_plan()
    if a.dependency_smoke:
        runtime=runtime_module();cmd=runtime.container_command(admission.read(AUTHOR_PLAN),[str(D/'worker.py'),'--preflight-only','--out',str(D/'preflight001.json')],False,'wavefront-dependency-smoke')
        if '--gpus' in cmd or any(x.startswith('/dev/nvidia') for x in cmd):raise ValueError('GPU in dependency smoke')
        print('actual cwd launcher dependencies loaded; no GPU or Docker used');return 0
    admission.check_review('preflight')
    if a.stage=='run':admission.check_review('final')
    initial=admission.read(B/'initial_plan.json')
    budget={'budget_source':str(B/'initial_plan.json'),'budget_source_sha256':p['budget_source_sha256'],
            'budget_start_unix':initial['budget_start_unix'],'budget_end_unix':initial['budget_end_unix']}
    inner,outer,grace=(600,720,90) if a.stage=='preflight' else (10500,10920,90)
    validate_budget_window(budget,time.time(),inner,outer,grace)
    mon=D/('monitor_'+a.stage+'001');started=admission.read(mon/'started.json')
    if started.get('worker_pid')!=os.getpid() or started.get('timeout_seconds')!=outer or started.get('grace_seconds')!=grace:
        raise ValueError('supervisor mismatch')
    if a.stage=='run' and (started.get('stale_seconds')!=1200 or started.get('progress_paths')!=[str(D/'run_run001/observations/progress.jsonl')]):
        raise ValueError('stale supervision mismatch')
    if shutil.disk_usage(D).free<32*1024**3:raise ValueError('insufficient disk')
    runtime=runtime_module()
    if a.stage=='run' and runtime.gpu_pids():raise ValueError('GPU is busy')
    out=D/('run_'+a.stage+'001');out.mkdir(exist_ok=False)
    name='eadp-wavefront-full-'+a.stage+'-'+str(os.getpid())
    worker=[str(D/'worker.py')]
    if a.stage=='preflight':worker+=['--preflight-only','--out',str(D/'preflight001.json')]
    else:worker+=['--out',str(out/'observations')]
    cmd=runtime.container_command(admission.read(AUTHOR_PLAN),worker,a.stage=='run',name)
    if cmd.count('--rm')!=1:raise ValueError('unexpected container command')
    cmd.remove('--rm')
    with (out/'launch.json').open('x') as f:
        import json
        json.dump({'stage':a.stage,'plan_sha256':admission.sha(D/'plan002.json'),'command':cmd,
                   'container_name':name,'controller':runtime.identity(os.getpid()),'started_unix':time.time()},f,indent=2)
        f.write('\n');f.flush();os.fsync(f.fileno())
    return run_bounded(cmd,name,inner,runtime.gpu_pids,runtime.identity,out)

if __name__=='__main__':sys.exit(main())
