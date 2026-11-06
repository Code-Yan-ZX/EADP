"""Candidate sequential two-container identity controller; gate unfrozen."""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PROJECT=ROOT.parents[2];GATE_SHA='1468583a79dd55772be0ec268517d90ab9ecfb4e821bc28656a1c9f217f75819'
def load(name,path,sha):
 if name in sys.modules or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('module identity')
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def write(path,obj):
 with path.open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def dispatch(runtime,timing,author,run,approval,read):
 run=Path(run);run.mkdir();states=[]
 for arm in ('rres','eadp'):
  before=runtime.gpu_pids()
  if before!=[]:raise ValueError('GPU busy before diagnostic arm')
  armdir=run/arm;armdir.mkdir();out=armdir/'observations';out.mkdir();name='eadp-rres-identity3-'+arm+'-'+str(os.getpid());command=runtime.container_command(author,[str(HERE/'gpu_worker.py'),'--arm',arm,'--out',str(out)],True,name)
  devices=[command[i+1] for i,x in enumerate(command[:-1]) if x=='--device'];envs=[command[i+1] for i,x in enumerate(command[:-1]) if x=='--env']
  if command.count('--rm')!=1 or '--gpus' in command or sorted(devices)!=sorted(['/dev/nvidia0','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools']) or envs.count('CUDA_VISIBLE_DEVICES=0')!=1:raise ValueError('container device command')
  command.remove('--rm');write(armdir/'launch.json',{'approval':approval,'controller':runtime.identity(os.getpid()),'command':command,'name':name,'arm':arm,'gpu_pids_before':before})
  rc=timing.run_bounded(command,name,1600,runtime.gpu_pids,runtime.identity,armdir,require_gpu_idle=True)
  if rc:raise RuntimeError('diagnostic container failed: '+arm)
  result=read(out/'worker_complete.json');states.append({'arm':arm,'status':result['status']})
 status='passed' if len(states)==2 and all(x['status']=='passed' for x in states) else 'failed';write(run/'controller_complete.json',{'status':status,'arms':states,'GPU_authorized':True,'GT_opened':False,'timing_claims':False})
 return 0 if status=='passed' else 1
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',choices=['001'],required=True);p.parse_args();gate=load('_identity_gpu_host_gate',HERE/'gpu_gate.py',GATE_SHA);approval=gate.admit();s=gate.read(HERE/'monitor_run001/started.json');own=[x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x];parent=[x.decode() for x in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().split(b'\0') if x]
 progress=[str((HERE/'run001'/arm/'observations/progress.jsonl').resolve()) for arm in ('rres','eadp')]
 if s.get('worker_pid')!=os.getpid() or os.getpid()!=os.getpgrp() or os.getpid()!=os.getsid(0) or s.get('command')!=own or Path(s.get('cwd','')).resolve()!=ROOT or s.get('timeout_seconds')!=3600 or s.get('stale_seconds')!=1200 or s.get('grace_seconds')!=120 or s.get('progress_paths')!=progress or not any((ROOT/x).resolve()==ROOT/'monitor/supervise.py' for x in parent):raise ValueError('supervisor identity')
 if time.time()+3720>=gate.read(ROOT/'initial_plan.json')['budget_end_unix']:raise ValueError('research deadline')
 sources=gate.read(HERE/'gpu_plan001.json')['source_sha256'];guardpath=ROOT/'e6/pixel_wavefront_model_t020_run_v2/imports_guard.py';guard=load('_identity_gpu_guard',guardpath,sources[str(guardpath)]);sys.path.insert(0,str(ROOT/'e4_c_latency'));guard.check(guard.LATENCY);import timing_launch;guard.check(guard.LATENCY,require=True)
 path=ROOT/'e6/pixel_wavefront_t020_timing_final_v9/preflight_launch.py';loader=load('_identity_gpu_author_loader',path,sources[str(path)]);runtime=loader.load_author(sources);author=gate.read(PROJECT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json')
 return dispatch(runtime,timing_launch,author,HERE/'run001',approval,gate.read)
if __name__=='__main__':raise SystemExit(main())
