"""Bounded CPU-only Docker test using the actual launcher environment builder."""
import json,os,signal,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[2];PLAN=BASE.parents[1]/'20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
sys.path.insert(0,str(PLAN.parent))
from run_panel import container_command
from smoke_launch import configure_container
plan=json.loads(PLAN.read_text());cache=HERE/'container_preflight002_cache';cache.mkdir(exist_ok=False)
name='eadp-e6prior-cpu-preflight002-'+str(os.getpid())
cmd=container_command(plan,[str(HERE/'container_preflight002.py')],False,name)
idx=cmd.index('--entrypoint');cmd[idx+1]=str(HERE.parent/'install001/env/bin/python')
cmd=configure_container(cmd,cache);at=cmd.index('--workdir');cmd[at:at]=['--env','CUDA_VISIBLE_DEVICES=']
if '--gpus' in cmd:raise ValueError('preflight must not request GPU')
with (HERE/'container_preflight002_launch.json').open('x') as f:json.dump({'command':cmd,'scope':'CPU only'},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def interrupted(_sig,_frame):raise KeyboardInterrupt('external termination')
for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
try:
 p=subprocess.run(cmd,text=True,capture_output=True,timeout=600)
 (HERE/'container_preflight002.stdout').write_text(p.stdout);(HERE/'container_preflight002.stderr').write_text(p.stderr)
 if p.returncode!=0:raise RuntimeError('preflight failed with exit '+str(p.returncode))
 result=json.loads(p.stdout.strip().splitlines()[-1])
 if result.get('status')!='passed' or result.get('cuda_initialized') is not False:raise ValueError('preflight proof failed')
 print(json.dumps(result))
finally:
 subprocess.run(['docker','stop','--time','5',name],capture_output=True,text=True,timeout=15,check=False)
