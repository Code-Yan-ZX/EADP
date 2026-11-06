"""Bounded retained no-device Docker for execution-chain imports only."""
import argparse,importlib.util,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PROJECT=ROOT.parents[2]
def load(name,path):
 if name in sys.modules:raise ValueError('preloaded module')
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);a=ap.parse_args();out=a.out.resolve()
 if out!=(HERE/'cpu_preflight001').resolve() or out.exists():raise ValueError('unplanned output')
 import cpu_gate,gpu_gate
 approval=cpu_gate.admit();started=gpu_gate.read(HERE/'monitor_cpu_preflight001/started.json');own=[x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x];parent=[x.decode() for x in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().split(b'\0') if x]
 if started.get('worker_pid')!=os.getpid() or os.getpid()!=os.getpgrp() or os.getpid()!=os.getsid(0) or started.get('command')!=own or Path(started.get('cwd','')).resolve()!=ROOT or started.get('timeout_seconds')!=900 or started.get('grace_seconds')!=60 or not any((ROOT/x).resolve()==ROOT/'monitor/supervise.py' for x in parent):raise ValueError('supervisor identity')
 if time.time()+960>=gpu_gate.read(ROOT/'initial_plan.json')['budget_end_unix']:raise ValueError('research deadline')
 source=cpu_gate.plan()['source_sha256'];guard=load('_identity_execution_cpu_guard',ROOT/'e6/pixel_wavefront_model_t020_run_v2/imports_guard.py');sys.path.insert(0,str(ROOT/'e4_c_latency'));guard.check(guard.LATENCY);import timing_launch;guard.check(guard.LATENCY,require=True)
 loader=load('_identity_execution_cpu_author',ROOT/'e6/pixel_wavefront_t020_timing_final_v9/preflight_launch.py');runtime=loader.load_author(source);author=gpu_gate.read(PROJECT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json');out.mkdir();name='eadp-identity3-exec-cpu-'+str(os.getpid());command=runtime.container_command(author,[str(HERE/'cpu_entry.py'),'--out',str(out)],False,name)
 if command.count('--rm')!=1 or '--gpus' in command or '--device' in command:raise ValueError('device exposure')
 command.remove('--rm')
 with (out/'launch.json').open('x') as f:json.dump({'approval':approval,'controller':runtime.identity(os.getpid()),'command':command,'name':name},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return timing_launch.run_bounded(command,name,750,None,runtime.identity,out,require_gpu_idle=False)
if __name__=='__main__':raise SystemExit(main())
