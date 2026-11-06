"""Real no-device bridge/import preflight; no model creation or GPU call."""
import argparse,importlib.util,json,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent
def load(name,path):
 p=Path(path).resolve();old=sys.modules.get(name)
 if old is not None:
  if not getattr(old,'__file__',None) or Path(old.__file__).resolve()!=p:raise ValueError('foreign module '+name)
  return old
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if out!=(D/'run_preflight001/worker_preflight.json').resolve() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('unplanned output or device')
 gate=load('_pixel_t020_runtime_preflight_gate',D/'preflight_gate.py');approval=gate.preflight()
 bridge=load('_pixel_t020_runtime_bridge',D/'bridge.py');loaded=bridge.prepare_cpu();prepared=loaded['model'].prepare_cpu()
 if len(prepared['references'])!=6 or prepared['torch'].cuda.is_initialized():raise ValueError('preflight initialized CUDA or missed reference')
 try:gate.gpu()
 except RuntimeError:pass
 else:raise ValueError('GPU not locked')
 value={'status':'passed','approval':approval,'references':6,'module_paths':{k:str(Path(v.__file__).resolve()) for k,v in loaded.items()},'model_module_paths':prepared['module_paths'],'GPU_authorized':False,'model_loaded':False,'cuda_initialized':False,'GT_opened':False,'existing_DEV_predictions_opened':True}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
