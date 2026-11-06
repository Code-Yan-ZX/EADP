"""Actual execution-chain imports in no-device Docker; never runs GPU."""
import argparse,inspect,json,os,socket,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];IDENTITY=ROOT/'e6/rres_eadp_identity3_v2';TIMING=ROOT/'e6/rres_eadp_timing_pilot_v4'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve()
 if out!=(HERE/'cpu_preflight001').resolve() or not out.is_dir() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('unplanned noGPU entry')
 import cpu_gate,gpu_gate
 approval=cpu_gate.admit();deadline=time.monotonic()+60
 while True:
  try:i=gpu_gate.read(out/'container_identity.json');launch=gpu_gate.read(out/'launch.json');break
  except (FileNotFoundError,json.JSONDecodeError):
   if time.monotonic()>deadline:raise TimeoutError('identity publication')
   time.sleep(.2)
 cid=i.get('container_id');command=launch['command']
 if not isinstance(cid,str) or len(cid)!=64 or socket.gethostname()!=cid[:12] or cid not in Path('/proc/self/cgroup').read_text() or '--gpus' in command or '--device' in command or launch.get('approval')!=approval:raise ValueError('container identity')
 sys.path[:0]=[str(IDENTITY),str(TIMING)];import reference_inputs
 refs=reference_inputs.load()
 import gpu_worker,controller,snapshot,verify,docker_contract,torch
 modules=[gpu_worker,controller,snapshot,verify,docker_contract,cpu_gate,gpu_gate]
 if any(Path(inspect.getfile(m)).resolve()!=HERE/(m.__name__+'.py') for m in modules) or gpu_worker.GATE_SHA!=gpu_gate.sha(HERE/'gpu_gate.py') or controller.GATE_SHA!=gpu_worker.GATE_SHA or torch.cuda.is_initialized() or len(refs['rows'])!=62:raise ValueError('execution import identity')
 result={'status':'passed','scope':'execution chain imports only','container_id':cid,'module_paths':{m.__name__:str(Path(inspect.getfile(m)).resolve()) for m in modules},'inputs':62,'model_loaded':False,'cuda_initialized':False,'GPU_authorized':False,'GT_opened':False,'existing_predictions_opened':True,'errors':[]}
 with (out/'worker_preflight.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
