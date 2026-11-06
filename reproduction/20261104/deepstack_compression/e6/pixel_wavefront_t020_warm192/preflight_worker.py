"""Real no-device processor preflight inside the reviewed retained container."""
import argparse,hashlib,importlib.util,json,os,socket,subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];GATE_SHA='80cde09a2fd7426e2f793eaea57fcde9f988be95f8e9d7165c0a240dc80867f3'
def load(name,path,digest):
 p=Path(path).resolve()
 if name in sys.modules or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('cached or changed source')
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve()
 if out!=(D/'run_preflight001/worker_preflight.json').resolve() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('noGPU output/environment')
 gate=load('_warm192_cpu_gate',D/'gate.py',GATE_SHA);approval=gate.admit_cpu();plan=gate.read(D/'preflight_plan001.json');source=plan['source_sha256'];deadline=time.monotonic()+60;path=D/'run_preflight001/container_identity.json'
 while True:
  try:identity=gate.read(path);break
  except (FileNotFoundError,json.JSONDecodeError):
   if time.monotonic()>deadline:raise TimeoutError('container identity')
   time.sleep(.2)
 launch=gate.read(D/'run_preflight001/launch.json');cid=identity.get('container_id','')
 if len(cid)!=64 or socket.gethostname()!=cid[:12] or cid not in Path('/proc/self/cgroup').read_text() or launch.get('approval')!=approval or '--gpus' in launch['command'] or '--device' in launch['command']:raise ValueError('not approved no-device container')
 worker=(D/'worker.py').resolve()
 if gate.sha(worker)!=source[str(worker)]:raise ValueError('worker source')
 smoke=subprocess.run([sys.executable,str(worker),'--no-gpu-dependency-smoke'],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=300,check=True);result=json.loads(smoke.stdout)
 if result.get('status')!='passed' or result.get('actual_inputs')!=193 or result.get('cuda_initialized') is not False or result.get('model_loaded') is not False:raise ValueError('actual worker import smoke')
 impl=load('_warm192_preflight_impl',D/'preflight_impl.py',source[str((D/'preflight_impl.py').resolve())]);model=load('_warm192_preflight_model',D/'model.py',source[str((D/'model.py').resolve())]);value=impl.run(gate.read(D/'design001.json'),source,model,out)
 if value.get('inputs_checked')!=193:raise ValueError('processor count')
 return 0
if __name__=='__main__':raise SystemExit(main())
