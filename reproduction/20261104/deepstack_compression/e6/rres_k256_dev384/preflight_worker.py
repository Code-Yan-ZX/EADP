"""Bounded no-device processor and weight verification inside actual container."""
import argparse,hashlib,importlib.util,json,os,socket,subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];ROOT=B.parents[2];GATE_SHA='192a88473d9c095af27aadf341af5c7ec57e52bc0e83d134892d1976931e448a'
def load_gate():
 path=D/'preflight_gate.py'
 if hashlib.sha256(path.read_bytes()).hexdigest()!=GATE_SHA or '_rres384_gate' in sys.modules:raise ValueError('gate identity')
 sp=importlib.util.spec_from_file_location('_rres384_gate',path);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);return m
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();run=D/'run_preflight001';out=a.out.resolve()
 if out!=run/'worker_preflight.json' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('noGPU scope')
 sys.path.insert(0,str(D));gate=load_gate();approval=gate.admit_preflight();plan=gate.read(D/'preflight_plan001.json');deadline=time.monotonic()+60
 while True:
  try:identity=gate.read(run/'container_identity.json');break
  except (FileNotFoundError,json.JSONDecodeError):
   if time.monotonic()>deadline:raise TimeoutError('container identity publication')
   time.sleep(.2)
 launch=gate.read(run/'launch.json');cid=identity['container_id']
 if len(cid)!=64 or socket.gethostname()!=cid[:12] or cid not in Path('/proc/self/cgroup').read_text() or launch.get('approval')!=approval or '--device' in launch['command'] or '--gpus' in launch['command']:raise ValueError('noGPU container identity')
 weights=gate.read(ROOT/'reproduction/20261028/qwen/qwen_weight_identity.json')['files'];stats=[]
 if len(weights)!=4:raise ValueError('weight inventory')
 for item in weights:
  path=Path(item['path']);before=path.stat();a=lambda s:[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
  if before.st_size!=item['bytes'] or gate.sha(path)!=item['expected_lfs_sha256'] or a(before)!=a(path.stat()):raise ValueError('weight payload')
  stats.append({'path':str(path),'stat':a(path.stat()),'sha256':item['expected_lfs_sha256']})
 probe=run/'probe';probe.mkdir();subprocess.run([sys.executable,str(D/'probe_processor.py'),'--out',str(probe)],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),timeout=4000,check=True)
 result=gate.read(probe/'processor_probe.json')
 if result.get('status')!='passed' or result.get('images')!=384 or result.get('model_loaded') is not False or result.get('cuda_initialized') is not False or result.get('GT_opened') is not False or any(gate.sha(k)!=v for k,v in plan['source_sha256'].items()):raise ValueError('incomplete probe or changed sources')
 if any([s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=item['stat'] for item in stats for s in [Path(item['path']).stat()]):raise ValueError('weight stat changed')
 with out.open('x') as f:json.dump({'status':'passed','approval':approval,'container_id':cid,'processor_probe_sha256':gate.sha(probe/'processor_probe.json'),'weights':stats,'images':384,'visual_tokens':256,'model_loaded':False,'cuda_initialized':False,'GPU_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'scope':'noGPU processor and weights; GPU remains locked','errors':[]},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
