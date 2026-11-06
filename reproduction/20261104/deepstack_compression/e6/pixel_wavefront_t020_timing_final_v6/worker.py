"""Prospective real model worker. No plan/proof exists; GPU remains locked."""
import argparse,hashlib,importlib.util,json,os,socket,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];R=B/'e6/pixel_wavefront_t020_timing_runtime_v4';DESIGN=B/'e6/pixel_wavefront_t020_model_timing_v2/design001.json'
# Must be set to the exact reviewed gate source digest in the final frozen copy.
GATE_SHA='0287d21d559c622902531de74c87b80cd976368ec93698aea41409342dfbcd6d'
def load(name,path,expected):
 p=Path(path).resolve()
 if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('source mismatch before import')
 m=sys.modules.get(name)
 if m is not None:
  if not getattr(m,'__file__',None) or Path(m.__file__).resolve()!=p:raise ValueError('foreign module '+name)
  return m
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m
 try:sp.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--job-index',type=int,required=True);a=parser.parse_args()
 gate=load('_pixel_t020_final_gpu_gate',D/'gpu_gate.py',GATE_SHA);approval=gate.admit()
 p=gate.read(D/'gpu_plan001.json');source=p['source_sha256'];raw=DESIGN.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=approval['design_sha256']:raise ValueError('design mismatch')
 jobs=json.loads(raw)['jobs']
 if jobs!=approval['jobs'] or a.job_index not in range(len(jobs)):raise ValueError('schedule mismatch')
 job=jobs[a.job_index];d=D/'run_001'/('job_%03d'%a.job_index);launch=gate.read(d/'launch.json');cmd=launch.get('command',[])
 if cmd.count(str(D/'worker.py'))!=1 or cmd.count('--job-index')!=1 or cmd[cmd.index('--job-index')+1:cmd.index('--job-index')+2]!=[str(a.job_index)]:raise ValueError('worker argv')
 bridge=load('_pixel_t020_final_bridge',R/'bridge.py',source[str((R/'bridge.py').resolve())]);parts=bridge.prepare_cpu()
 release=parts['worker'].wait_release(job,d,launch,socket.gethostname(),approval['plan_sha256'])
 if release['container_id'] not in Path('/proc/self/cgroup').read_text():raise ValueError('self cgroup mismatch')
 try:return bridge.unsafe_run(job,d,parts)
 except BaseException as exc:
  error=d/'worker_exception.json'
  with error.open('x') as f:json.dump({'job':job,'error_type':type(exc).__name__,'error':str(exc),'raw_exists':(d/'raw.json').exists()},f);f.write('\n');f.flush();os.fsync(f.fileno())
  raise
if __name__=='__main__':main()
