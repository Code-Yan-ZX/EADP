"""Candidate two-arm diagnostic worker. Gate digest intentionally unfrozen."""
import argparse,hashlib,importlib.util,json,os,socket,sys,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PROJECT=ROOT.parents[2];IDENTITY=ROOT/'e6/rres_eadp_identity3_v2';TIMING=ROOT/'e6/rres_eadp_timing_pilot_v4'
GATE_SHA='1468583a79dd55772be0ec268517d90ab9ecfb4e821bc28656a1c9f217f75819'
def load(name,path,expected):
 if name in sys.modules or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('gate source or module identity')
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def read(p):return json.loads(Path(p).read_text())
def weight_stats():
 prior=read(ROOT/'e6/rres_k256_dev384/run_preflight001/worker_preflight.json')['weights'];result=[]
 for w in prior:
  s=Path(w['path']).stat();result.append({'path':w['path'],'stat':[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns],'sha256':w['sha256']})
 if result!=prior:raise ValueError('weight endpoint drift')
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['rres','eadp'],required=True);ap.add_argument('--out',required=True,type=Path);a=ap.parse_args();out=a.out.resolve()
 if out!=(HERE/'run001'/a.arm/'observations').resolve() or not out.is_dir():raise ValueError('output path')
 gate=load('_rres_identity_gpu_gate_worker',HERE/'gpu_gate.py',GATE_SHA);approval=gate.admit();run=out.parent;deadline=time.monotonic()+60
 while True:
  try:identity=read(run/'container_identity.json');launch=read(run/'launch.json');break
  except (FileNotFoundError,json.JSONDecodeError):
   if time.monotonic()>deadline:raise TimeoutError('container identity')
   time.sleep(.2)
 cid=identity.get('container_id')
 if not isinstance(cid,str) or len(cid)!=64 or socket.gethostname()!=cid[:12] or cid not in Path('/proc/self/cgroup').read_text() or launch.get('approval')!=approval or launch.get('arm')!=a.arm:raise ValueError('GPU container identity')
 before=weight_stats();sys.path[:0]=[str(IDENTITY),str(TIMING)]
 import reference_inputs
 prepared=reference_inputs.load()
 for task,qid in (('DocVQA_VAL','64588'),('TextVQA_VAL','36511'),('AI2D_TEST','207406')):
  if (task,qid) not in prepared['rows'] or (a.arm,task,qid) not in prepared['references']:raise ValueError('missing admitted identity input or reference')
 import torch
 if torch.cuda.is_initialized() or torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('device mismatch')
 worker=load('_rres_identity_actual_worker',IDENTITY/'worker.py',gate.read(HERE/'gpu_plan001.json')['source_sha256'][str((IDENTITY/'worker.py').resolve())]);result=worker.execute_after_future_admission(arm=a.arm,prepared=prepared,out=out)
 after=weight_stats()
 with (out/'identity_endpoints.json').open('x') as f:json.dump({'status':'passed','weights_before':before,'weights_after':after,'source_plan_sha256':gate.sha(HERE/'gpu_plan001.json'),'model_result':result['status'],'GT_opened':False},f);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0
if __name__=='__main__':
 try:raise SystemExit(main())
 except Exception as exc:
  if '--out' in sys.argv and '--arm' in sys.argv:
   try:
    out=Path(sys.argv[sys.argv.index('--out')+1]).resolve();arm=sys.argv[sys.argv.index('--arm')+1]
    if arm in ('rres','eadp') and out==(HERE/'run001'/arm/'observations').resolve() and out.is_dir():
     with (out/'entry_failure.json').open('x') as f:json.dump({'status':'failed','arm':arm,'failure_type':type(exc).__name__,'failure':str(exc),'traceback':traceback.format_exc(),'GT_opened':False},f);f.write('\n');f.flush();os.fsync(f.fileno())
   except Exception:pass
  raise
