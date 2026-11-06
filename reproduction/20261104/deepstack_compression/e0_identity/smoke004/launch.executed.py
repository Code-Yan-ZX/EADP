"""Bounded one-shot E0 smoke launcher. No automatic retry."""
import json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path('/media/disk2/YZX/research/EADP_v2')
HERE=Path(__file__).resolve().parent
OLD=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02'
sys.path.insert(0,str(OLD))
from run_panel import container_command, gpu_pids, identity
from diagnose import sha
if gpu_pids(): raise RuntimeError('GPU compute process already exists')
plan=json.loads((OLD/'run_plan.json').read_text())
run_id = sys.argv[1] if len(sys.argv) in (2, 3) else None
if not run_id or not run_id.isalnum(): raise ValueError('pass unique alphanumeric run id')
limit = int(sys.argv[2]) if len(sys.argv) == 3 else 1
if not 1 <= limit <= 16: raise ValueError('limit must be 1..16')
out=HERE/run_id
if out.exists(): raise RuntimeError('refusing overwrite')
name='eadp-e0-'+run_id+'-'+str(os.getpid())
cmd=container_command(plan,[str(HERE/'worker.py'),'--out',str(out),'--limit',str(limit)],True,name)
launch=dict(command=cmd, controller=identity(os.getpid()), started_unix=time.time(),
    prior_gpu_pids=gpu_pids(), timeout_seconds=900,
    source_sha256={str(p):sha(p) for p in (HERE/'worker.py',HERE/'adapter.py',HERE/'launch.py')})
with (HERE/(run_id+'_launch.json')).open('x') as f: json.dump(launch,f,indent=2)
print('starting',name,flush=True)
with (HERE/(run_id+'.log')).open('x') as log:
    try:
        p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        result=dict(returncode=p.returncode,timed_out=False)
    except subprocess.TimeoutExpired:
        subprocess.run(['docker','stop','--time','10',name],check=False,stdout=log,stderr=log,timeout=30)
        result=dict(returncode=None,timed_out=True)
result.update(finished_unix=time.time(),gpu_pids_after=gpu_pids(),name=name,controller=identity(os.getpid()))
with (HERE/(run_id+'_exit.json')).open('x') as f: json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
raise SystemExit(result['returncode'] if result['returncode'] is not None else 124)
