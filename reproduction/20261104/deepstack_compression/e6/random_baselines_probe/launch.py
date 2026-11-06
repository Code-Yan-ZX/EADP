"""Bounded retained-container CPU synthetic probe."""
import importlib.util,json,os,signal,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def identity(pid):
 p=Path('/proc')/str(pid);return {'pid':pid,'start_ticks':int((p/'stat').read_text().split(') ')[1].split()[19])}
def main():
 old=ROOT/'m3_gpu_smoke/preflight_001/launch.json';cmd=json.loads(old.read_text())['command'][:]
 name='eadp-random-cpu-probe-001';cmd[cmd.index('--name')+1]=name
 i=cmd.index(str(ROOT/'m3_gpu_smoke/worker.py'));cmd=cmd[:i]+[str(HERE/'worker.py')]
 assert 'CUDA_VISIBLE_DEVICES=' in cmd and not any('/dev/nvidia' in x for x in cmd)
 out=HERE/'probe_001';out.mkdir(exist_ok=False)
 with (out/'launch.json').open('x') as f:json.dump({'command':cmd,'base':str(old)},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 interrupted=[None]
 def stop(signum,_frame):interrupted[0]=signum
 for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
 supervisor=load('_random_probe_supervisor',ROOT/'e6/selected_smoke/docker_supervisor.py')
 ex=supervisor.supervise(cmd,name,out,identity(os.getpid()),300,lambda:[],identity,interrupted)
 try:result=json.loads((out/'launcher.log').read_text().splitlines()[-1])
 except (ValueError,IndexError):result={}
 ok=(ex.get('returncode')==0 and ex.get('container_exit_code')==0 and ex.get('container_stopped_verified') is True and ex.get('client_exited_while_container_running') is False and ex.get('timed_out') is False and ex.get('interrupted') is None and not ex.get('errors') and result.get('status')=='passed')
 with (out/'verification.json').open('x') as f:json.dump({'status':'passed' if ok else 'failed','result':result,'scope':'synthetic CPU probe only; no model or generation'},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0 if ok else 1
if __name__=='__main__':sys.exit(main())
