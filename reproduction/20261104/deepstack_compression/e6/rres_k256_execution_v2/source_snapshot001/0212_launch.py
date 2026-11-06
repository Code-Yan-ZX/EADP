"""Bounded retained same-image no-GPU execution-import container."""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2];GATE_SHA='a679ea2f405193b0602ad33d7bab80a7d9df2a1e5ff5fc33f23d1ceb051f9328'
def load(name,path,digest):
 if name in sys.modules or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('source or preloaded module')
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if out!=D/'run001' or out.exists():raise ValueError('output')
 g=load('_rres_runtime_host_gate',D/'gate.py',GATE_SHA);approval=g.admit();s=g.read(D/'monitor_run001/started.json');cwd=Path(s['cwd']);cmd=s['command'];own=[x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x];parent=[x.decode() for x in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().split(b'\0') if x]
 if cwd.resolve()!=B or s['worker_pid']!=os.getpid() or os.getpgrp()!=os.getpid() or os.getsid(0)!=os.getpid() or cmd!=own or s['timeout_seconds']!=720 or s['grace_seconds']!=60 or cmd[-2:]!=['--out',str(out)] or not any((cwd/x).resolve()==B/'monitor/supervise.py' for x in parent):raise ValueError('supervision')
 if time.time()+780>=g.read(B/'initial_plan.json')['budget_end_unix']:raise ValueError('deadline')
 source=g.read(D/'plan001.json')['source_sha256'];guardpath=B/'e6/pixel_wavefront_model_t020_run_v2/imports_guard.py';guard=load('_rres_runtime_import_guard',guardpath,source[str(guardpath)]);guard.check(guard.LATENCY);sys.path.insert(0,str(B/'e4_c_latency'));import timing_launch;guard.check(guard.LATENCY,require=True)
 loaderpath=B/'e6/pixel_wavefront_t020_timing_final_v9/preflight_launch.py';loader=load('_rres_runtime_author_loader',loaderpath,source[str(loaderpath)]);runtime=loader.load_author(source);author=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';out.mkdir();name='eadp-rres-runtime-cpu-'+str(os.getpid());command=runtime.container_command(g.read(author),[str(D/'worker.py'),'--out',str(out/'worker.json')],False,name)
 if command.count('--rm')!=1 or '--gpus' in command or '--device' in command:raise ValueError('GPU device')
 command.remove('--rm')
 with (out/'launch.json').open('x') as f:json.dump({'approval':approval,'controller':runtime.identity(os.getpid()),'command':command,'name':name},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return timing_launch.run_bounded(command,name,600,None,runtime.identity,out,require_gpu_idle=False)
if __name__=='__main__':raise SystemExit(main())
