"""No-device CPU import preflight; no model load and no timing."""
import argparse,importlib.util,json,os,sys
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];C=B/'e6/pixel_wavefront_t020_model_timing_v2'
def load(name,path):
 path=Path(path).resolve();old=sys.modules.get(name)
 if old is not None:
  if not getattr(old,'__file__',None) or Path(old.__file__).resolve()!=path:raise ValueError('foreign module '+name)
  return old
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if out!=(D/'run_preflight001/worker_preflight.json').resolve() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('unexpected preflight path/device')
 gate=load('_pixel_t020_timing_execution_gate',D/'gate.py');approval=gate.preflight();m=load('_pixel_t020_timing_model_candidate',C/'model_candidate.py');prepared=m.prepare_cpu()
 if prepared['torch'].cuda.is_initialized() or len(prepared['references'])!=6:raise ValueError('CUDA/model/identity preflight')
 try:gate.gpu()
 except RuntimeError:pass
 else:raise ValueError('GPU unexpectedly admitted')
 value={'status':'passed','approval':approval,'identity':prepared['identity'],'module_paths':prepared['module_paths'],'references':len(prepared['references']),'GT_opened':False,'existing_DEV_predictions_opened':True,'model_loaded':False,'cuda_initialized':False,'gpu_authorized':False}
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
