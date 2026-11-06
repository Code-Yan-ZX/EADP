"""Same-image no-device processor and weight preflight; no model/GPU authority."""
import argparse,hashlib,importlib.util,json,os,socket,subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];GATE_SHA='e04bb27e28475900ede83e6f5761688ecb74befa1c5f06e3f5fe47b7d216f2df'
def load(name,path):
 if name in sys.modules or hashlib.sha256(path.read_bytes()).hexdigest()!=GATE_SHA:raise ValueError('gate source')
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();out=a.out.resolve();run=D/'run_preflight001'
 if out!=run/'worker_preflight.json' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('noGPU only')
 gate=load('_rres_preflight_gate',D/'gate.py');approval=gate.admit_preflight();plan=gate.read(D/'preflight_plan001.json');deadline=time.monotonic()+60
 sys.path.insert(0,str(D));import arm,capture,resolution
 for module,name in ((arm,'arm.py'),(capture,'capture.py'),(resolution,'resolution.py')):
  actual=Path(module.__file__).resolve()
  if actual!=D/name or gate.sha(actual)!=plan['source_sha256'][str(actual)]:raise ValueError('diagnostic import source')
 while True:
  try:identity=gate.read(run/'container_identity.json');break
  except (FileNotFoundError,json.JSONDecodeError):
   if time.monotonic()>deadline:raise TimeoutError('container identity')
   time.sleep(.2)
 launch=gate.read(run/'launch.json');cid=identity.get('container_id','')
 if len(cid)!=64 or socket.gethostname()!=cid[:12] or cid not in Path('/proc/self/cgroup').read_text() or launch.get('approval')!=approval or '--device' in launch['command'] or '--gpus' in launch['command']:raise ValueError('container identity or devices')
 weights=gate.read(B.parents[2]/'reproduction/20261028/qwen/qwen_weight_identity.json')['files'];stats={}
 if len(weights)!=4:raise ValueError('weight inventory')
 for item in weights:
  path=Path(item['path']);before=path.stat()
  if before.st_size!=item['bytes'] or gate.sha(path)!=item['expected_lfs_sha256']:raise ValueError('weight payload')
  after=path.stat();x=lambda s:[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
  if x(before)!=x(after):raise ValueError('weight changed while reading')
  stats[str(path)]=x(after)
 probe=run/'processor_probe.json';p=subprocess.run([sys.executable,str(D/'probe_processor.py'),'--out',str(probe)],cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True,timeout=360,check=True);result=gate.read(probe)
 if result.get('status')!='passed' or result.get('images')!=3 or result.get('model_loaded') is not False or result.get('cuda_initialized') is not False or result.get('GT_opened') is not False or result.get('GPU_authorized') is not False or any(gate.sha(k)!=v for k,v in plan['source_sha256'].items()):raise ValueError('processor/plan mismatch')
 with out.open('x') as f:json.dump({'status':'passed','approval':approval,'container_id':cid,'processor_probe_sha256':gate.sha(probe),'diagnostic_imports':['arm.py','capture.py','resolution.py'],'weights_stats':stats,'images':3,'visual_tokens':256,'model_loaded':False,'cuda_initialized':False,'GPU_authorized':False,'GT_opened':False,'existing_DEV_predictions_opened':True,'scope':'same-image noGPU processor and weight preflight; GPU execution still locked','errors':[]},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
