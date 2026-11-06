"""PID-1 tiny-allocation worker; GPU entry only via separately reviewed controller."""
import argparse,importlib.util,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent

def save(path,obj):
 with Path(path).open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def main():
 p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');p.add_argument('--out',required=True,type=Path);p.add_argument('--nonce');a=p.parse_args()
 sys.path.insert(0,str(D));import gate
 if Path(gate.__file__).resolve()!=D/'gate.py':raise ValueError('gate import collision')
 gate.verify_plan()
 import torch
 if torch.cuda.is_initialized():raise ValueError('CUDA unexpectedly initialized')
 spec=importlib.util.spec_from_file_location('eadp_memory_real_barrier',B/'e4_c_memory_runtime/worker_barrier.py');barrier=importlib.util.module_from_spec(spec);spec.loader.exec_module(barrier)
 if Path(barrier.__file__).resolve()!=B/'e4_c_memory_runtime/worker_barrier.py':raise ValueError('barrier source mismatch')
 if a.preflight_only:
  if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('preflight must hide GPU')
  save(a.out,{'status':'passed','cuda_initialized':False,'model_loaded':False,'GT_opened':False,'gpu_used':False,'torch':torch.__version__,'barrier_path':str(Path(barrier.__file__).resolve())});return
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or a.nonce is None or os.getpid()!=1:raise ValueError('requires reviewed GPU container PID1')
 gate.verify_review()
 expected,release=barrier.handshake(a.out,a.nonce,torch.cuda.is_initialized,timeout=60)
 if torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
 t=torch.ones((16*1024*1024,),dtype=torch.uint8,device='cuda');torch.cuda.synchronize()
 save(a.out/'allocation.json',{'allocation_bytes':int(t.numel()*t.element_size()),'allocation_monotonic_ns':time.monotonic_ns(),'gpu_model':torch.cuda.get_device_name(0),'status':'allocated','model_loaded':False,'GT_opened':False})
 time.sleep(8)
 free_start=time.monotonic_ns();del t;torch.cuda.empty_cache();torch.cuda.synchronize()
 save(a.out/'worker_complete.json',{'status':'completed','allocation_bytes':16*1024*1024,'free_start_monotonic_ns':free_start,'release_monotonic_ns':time.monotonic_ns(),'model_loaded':False,'GT_opened':False})
if __name__=='__main__':main()
