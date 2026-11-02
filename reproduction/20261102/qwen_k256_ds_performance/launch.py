import json, os, subprocess, sys, time
from pathlib import Path
r=Path('/media/disk2/YZX/research/EADP_v2');old=r/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02'
sys.path.insert(0,str(old))
from run_panel import container_command,gpu_pids,identity
assert gpu_pids()==[],gpu_pids()
p=json.loads((old/'run_plan.json').read_text())
here=Path(__file__).resolve().parent
cmd=container_command(p,[str(here/'benchmark.py')],True,'eadp-qwen-k256-ds-perf-20261102')
(here/'launch.json').write_text(json.dumps(dict(command=cmd,controller=identity(os.getpid()),started_unix=time.time(),gpu_compute_pids_before=gpu_pids()),indent=2)+'\n')
print('starting',flush=True)
with (here/'execution.log').open('x') as f:
 result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
(here/'process_exit.json').write_text(json.dumps(dict(container_exit_code=result.returncode,controller=identity(os.getpid()),finished_unix=time.time(),gpu_compute_pids_after=gpu_pids()),indent=2)+'\n')
print('container exit',result.returncode,flush=True)
sys.exit(result.returncode)
