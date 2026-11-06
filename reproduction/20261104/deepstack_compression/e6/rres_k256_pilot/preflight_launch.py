"""Reviewed bounded no-device container; GPU execution is not implemented."""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2];GATE_SHA='e04bb27e28475900ede83e6f5761688ecb74befa1c5f06e3f5fe47b7d216f2df'
def load(name,path,digest):
 if name in sys.modules or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('changed/preloaded module')
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if out!=D/'run_preflight001' or out.exists():raise ValueError('output')
 gate=load('_rres_host_gate',D/'gate.py',GATE_SHA);approval=gate.admit_preflight();started=gate.read(D/'monitor_preflight001/started.json');cmd=started['command'];cwd=Path(started['cwd']);own=[x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x];parent=[x.decode() for x in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().split(b'\0') if x]
 if cwd.resolve()!=B or started.get('worker_pid')!=os.getpid() or os.getpid()!=os.getpgrp() or os.getpid()!=os.getsid(0) or cmd!=own or started.get('timeout_seconds')!=1800 or started.get('grace_seconds')!=90 or cmd[-2:]!=['--out',str(out)] or not any((cwd/x).resolve()==B/'monitor/supervise.py' for x in parent):raise ValueError('supervisor identity')
 if time.time()+1890>=gate.read(B/'initial_plan.json')['budget_end_unix']:raise ValueError('budget')
 source=gate.read(D/'preflight_plan001.json')['source_sha256'];guardpath=B/'e6/pixel_wavefront_model_t020_run_v2/imports_guard.py';guard=load('_rres_import_guard',guardpath,source[str(guardpath)]);guard.check(guard.LATENCY);sys.path.insert(0,str(B/'e4_c_latency'));import timing_launch;guard.check(guard.LATENCY,require=True)
 loaderpath=B/'e6/pixel_wavefront_t020_timing_final_v9/preflight_launch.py';loader=load('_rres_author_loader',loaderpath,source[str(loaderpath)]);runtime=loader.load_author(source);author=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';out.mkdir();name='eadp-rres-k256-cpu-'+str(os.getpid());command=runtime.container_command(gate.read(author),[str(D/'preflight_worker.py'),'--out',str(out/'worker_preflight.json')],False,name)
 if command.count('--rm')!=1 or '--gpus' in command or '--device' in command:raise ValueError('GPU device')
 command.remove('--rm')
 with (out/'launch.json').open('x') as f:json.dump({'approval':approval,'controller':runtime.identity(os.getpid()),'command':command,'name':name},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return timing_launch.run_bounded(command,name,1650,None,runtime.identity,out,require_gpu_idle=False)
if __name__=='__main__':raise SystemExit(main())
