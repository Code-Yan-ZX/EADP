"""Bounded three-anchor GPU launcher. Actual execution requires both reviews."""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
from gpu_dispatch import dispatch
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2]
def load(name,path,sha):
 path=Path(path).resolve()
 if name in sys.modules or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('changed or preloaded module')
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-id',choices=['001'],required=True);ap.parse_args()
 gate=load('_rres_launch_gate',D/'gpu_gate.py','e825ffa729c0121f602e11990eb6645a775470798a55c5f2b1a00b44fa2a312b');approval=gate.admit_gpu();plan=gate.read(D/'gpu_plan001.json');s=gate.read(D/'monitor_run001/started.json');cmd=s['command'];cwd=Path(s['cwd']);own=[x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x];parent=[x.decode() for x in (Path('/proc')/str(os.getppid())/'cmdline').read_bytes().split(b'\0') if x]
 if cwd.resolve()!=B or s.get('worker_pid')!=os.getpid() or os.getpgrp()!=os.getpid() or os.getsid(0)!=os.getpid() or cmd!=own or s.get('timeout_seconds')!=3720 or s.get('grace_seconds')!=120 or cmd[-2:]!=['--run-id','001'] or not any((cwd/x).resolve()==B/'monitor/supervise.py' for x in parent):raise ValueError('supervisor identity')
 if time.time()+3840>=gate.read(B/'initial_plan.json')['budget_end_unix']:raise ValueError('project deadline')
 source=plan['source_sha256'];guardpath=B/'e6/pixel_wavefront_model_t020_run_v2/imports_guard.py';guard=load('_rres_gpu_import_guard',guardpath,source[str(guardpath)]);guard.check(guard.LATENCY);sys.path.insert(0,str(B/'e4_c_latency'));import timing_launch;guard.check(guard.LATENCY,require=True)
 loaderpath=B/'e6/pixel_wavefront_t020_timing_final_v9/preflight_launch.py';loader=load('_rres_gpu_author_loader',loaderpath,source[str(loaderpath)]);runtime=loader.load_author(source);runtime.approval=approval;author=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json'
 return dispatch(runtime,timing_launch,gate.read(author),D/'run001',D/'worker.py')

if __name__=='__main__':raise SystemExit(main())
