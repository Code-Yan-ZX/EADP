"""Bounded retained no-device container; no model/GPU/NVML access."""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2]
GATE_SHA='b9300e39d4936abf337ec59c885664deb810723ee9cf4cd29034fd77ac008dc1'
def load(name,path,digest):
 path=Path(path).resolve()
 if name in sys.modules or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('preloaded or changed module')
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m
 try:s.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();out=a.out.resolve()
 if out!=(D/'run_preflight001').resolve():raise ValueError('unplanned output')
 gate=load('_pixel_nvml_host_gate',D/'gate.py',GATE_SHA);approval=gate.admit();plan=gate.read(D/'plan001.json');s=gate.read(D/'monitor_preflight001/started.json');cmd=s['command'];cwd=Path(s['cwd']);parent=[x.decode() for x in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().split(b'\0') if x];actual=[x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x]
 if cwd.resolve()!=B or s.get('worker_pid')!=os.getpid() or os.getpid()!=os.getpgrp() or os.getpid()!=os.getsid(0) or s.get('command')!=actual or s.get('timeout_seconds')!=600 or s.get('grace_seconds')!=60 or (cwd/cmd[1]).resolve()!=Path(__file__) or cmd[-2:]!=['--out',str(out)] or not any((cwd/x).resolve()==B/'monitor/supervise.py' for x in parent):raise ValueError('supervisor identity')
 if time.time()+600+60>=gate.read(B/'initial_plan.json')['budget_end_unix']:raise ValueError('budget')
 src=plan['source_sha256'];guardpath=B/'e6/pixel_wavefront_model_t020_run_v2/imports_guard.py';guard=load('_pixel_nvml_guard',guardpath,src[str(guardpath.resolve())]);guard.check(guard.LATENCY);sys.path.insert(0,str(B/'e4_c_latency'));import timing_launch;guard.check(guard.LATENCY,require=True)
 prior=B/'e6/pixel_wavefront_t020_timing_final_v9/preflight_launch.py';author_loader=load('_pixel_nvml_author_loader',prior,src[str(prior.resolve())]);runtime=author_loader.load_author(src);author=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';out.mkdir(exist_ok=False);name='eadp-pixel-nvml-cpu-'+str(os.getpid());command=runtime.container_command(gate.read(author),[str(D/'worker.py'),'--out',str(out/'worker_preflight.json')],False,name)
 if command.count('--rm')!=1 or '--gpus' in command or '--device' in command:raise ValueError('unexpected GPU devices')
 command.remove('--rm')
 with (out/'launch.json').open('x') as f:json.dump({'approval':approval,'command':command,'name':name,'controller':runtime.identity(os.getpid())},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return timing_launch.run_bounded(command,name,420,None,runtime.identity,out,require_gpu_idle=False)
if __name__=='__main__':raise SystemExit(main())
