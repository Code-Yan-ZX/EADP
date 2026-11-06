"""Prospective GPU worker; unconditional lock until new complete GPU gate."""
import importlib.util,json,os,socket,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];R=B/'e6/pixel_wavefront_t020_timing_runtime_v3';C=B/'e6/pixel_wavefront_t020_model_timing_v2'
def load(name,path):
 p=Path(path).resolve();m=sys.modules.get(name)
 if m is not None:
  if not getattr(m,'__file__',None) or Path(m.__file__).resolve()!=p:raise ValueError('foreign '+name)
  return m
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m

def execute_after_admission(index, out, plan_sha):
 """Unsafe: caller must verify independent GPU admission before entering."""
 design=json.loads((C/'design001.json').read_text());jobs=design['jobs']
 if type(index) is not int or index not in range(len(jobs)):raise ValueError('job index')
 job=jobs[index];d=Path(out).resolve()/('job_%03d'%index);launch=json.loads((d/'launch.json').read_text());cmd=launch.get('command',[])
 if cmd.count(str(D/'gpu_worker.py'))!=1 or cmd.count('--job-index')!=1 or cmd[cmd.index('--job-index')+1:cmd.index('--job-index')+2]!=[str(index)]:raise ValueError('worker argv')
 bridge=load('_pixel_t020_gpu_bridge',R/'bridge.py');parts=bridge.prepare_cpu()
 parts['worker'].wait_release(job,d,launch,socket.gethostname(),plan_sha)
 return bridge.unsafe_run(job,d,parts)

def main():raise RuntimeError('GPU locked: full plan, noGPU preflight and independent admission not ready')
if __name__=='__main__':main()
