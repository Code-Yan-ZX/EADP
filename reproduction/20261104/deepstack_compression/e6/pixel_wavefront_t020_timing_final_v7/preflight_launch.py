"""Bounded retained Docker preflight only; never requests a GPU."""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2]
GATE_SHA='093122fa065223639d1e9ad095d9382dc6bf5c5938a12bd3daa5be2b925c0ea9'
def check_gate(path):
 p=Path(path).resolve()
 if hashlib.sha256(p.read_bytes()).hexdigest()!=GATE_SHA:raise ValueError('preflight gate source mismatch')
 return p
def load(name,p):
 p=Path(p).resolve();m=sys.modules.get(name)
 if m is not None:
  if not getattr(m,'__file__',None) or Path(m.__file__).resolve()!=p:raise ValueError('foreign module')
  return m
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m
 try:sp.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);a=ap.parse_args();out=a.out.resolve()
 if out!=(D/'run_preflight001').resolve():raise ValueError('unplanned output')
 g=load('_pixel_t020_final_preflight_gate',check_gate(D/'preflight_gate.py'));approval=g.preflight();started=g.read(D/'monitor_preflight001/started.json');cwd=Path(started['cwd']);cmd=started['command'];parent=os.getppid();parent_cmd=[x.decode() for x in (Path('/proc')/str(parent)/'cmdline').read_bytes().split(b'\0') if x];self_cmd=[x.decode() for x in Path('/proc/self/cmdline').read_bytes().split(b'\0') if x]
 if (started['worker_pid']!=os.getpid() or started['process_group']!=os.getpgrp() or os.getpid()!=os.getpgrp() or os.getpid()!=os.getsid(0) or started['command']!=self_cmd or (cwd/cmd[1]).resolve()!=Path(__file__) or cmd[-2:]!=['--out',str(out)] or started['timeout_seconds']!=900 or started['grace_seconds']!=90 or not any((cwd/x).resolve()==B/'monitor/supervise.py' for x in parent_cmd)):raise ValueError('supervisor process or command')
 sys.path.insert(0,str(B/'e4_c_latency'));guard=load('_pixel_t020_timing_import_guard',B/'e6/pixel_wavefront_model_t020_run_v2/imports_guard.py');guard.check(guard.LATENCY)
 import timing_launch
 guard.check(guard.LATENCY,require=True);initial=g.read(B/'initial_plan.json');budget={'budget_source':str(B/'initial_plan.json'),'budget_source_sha256':g.sha(B/'initial_plan.json'),'budget_start_unix':initial['budget_start_unix'],'budget_end_unix':initial['budget_end_unix']};timing_launch.validate_budget_window(budget,time.time(),720,900,90)
 author=ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_plan.json';runtime=load('_pixel_t020_timing_author_runtime',author.parent/'run_panel.py');out.mkdir(exist_ok=False);name='eadp-wavefront-t020-final-cpu-'+str(os.getpid());argv=runtime.container_command(g.read(author),[str(D/'preflight_entry.py'),'--out',str(out/'worker_preflight.json')],False,name)
 if argv.count('--rm')!=1 or '--gpus' in argv or '--device' in argv:raise ValueError('unexpected GPU devices')
 argv.remove('--rm')
 with (out/'launch.json').open('x') as f:json.dump({'approval':approval,'controller':runtime.identity(os.getpid()),'command':argv,'name':name},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return timing_launch.run_bounded(argv,name,720,None,runtime.identity,out,require_gpu_idle=False)
if __name__=='__main__':raise SystemExit(main())
