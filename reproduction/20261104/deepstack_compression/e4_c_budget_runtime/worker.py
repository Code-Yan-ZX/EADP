"""Full model-before-load worker path; real GPU entry is unconditionally locked."""
import argparse,importlib.util,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;V2=BASE/'e4_c_budget_pilot_v2'
def load_gate():
 name='_budget_runtime_gate';path=HERE/'gate.py';old=sys.modules.get(name)
 if old is not None:
  if Path(getattr(old,'__file__','')).resolve()!=path:raise ValueError('foreign gate')
  return old
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module
 try:spec.loader.exec_module(module)
 except BaseException:del sys.modules[name];raise
 return module
gate=load_gate()

def prepare(out):
 """Repeat all source, accepted input and weight checks without model/GPU."""
 prior=gate.validate_prior();runtime=gate.validate_runtime_sources();out=Path(out).resolve()
 if not out.is_dir() or (out/'processor_preflight.json').exists():raise ValueError('reused/missing output')
 spec=importlib.util.spec_from_file_location('_budget_v2_worker_preflight',V2/'preflight.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 saved=sys.argv
 try:
  sys.argv=[str(V2/'preflight.py'),'--out',str(out/'processor_preflight.json')];module.main()
 finally:sys.argv=saved
 result=json.loads((out/'processor_preflight.json').read_text())
 if result.get('cuda_initialized') is not False or result.get('model_loaded') is not False or result.get('status')!='passed':raise ValueError('CPU worker preflight failed')
 gate.validate_runtime_sources()
 record={'status':'passed','scope':'actual worker model-before-load path; GPU remains locked','processor_preflight_sha256':gate.sha(out/'processor_preflight.json'),
         'runtime_plan_sha256':gate.sha(HERE/'runtime_plan001.json'),'runtime_sources':runtime['source_sha256'],
         'prior_proof_sha256':prior['prior_proof_sha256'],'GT_opened':False,'scoring_performed':False,'predictions_opened':True,
         'prediction_scope':'accepted E1 mixed DEV journal and E4 DEV outputs for input and same-arm identity','model_loaded':False,'cuda_initialized':False}
 with (out/'worker_preflight.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 return record

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');args=p.parse_args()
 if not args.preflight_only:gate.admit_gpu()
 prepare(args.out)
if __name__=='__main__':main()
