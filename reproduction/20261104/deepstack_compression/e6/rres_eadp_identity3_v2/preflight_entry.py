"""Identity diagnostic actual imports; no model, CUDA or GPU execution."""
import argparse,inspect,json,os,socket,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PROJECT=ROOT.parents[2];TIMING=ROOT/'e6/rres_eadp_timing_pilot_v4'
def read(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args();out=args.out.resolve()
 if out!=(HERE/'preflight001').resolve() or not out.is_dir() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('unplanned or device-enabled probe')
 import gate
 approval=gate.preflight();deadline=time.monotonic()+60
 while True:
  try:identity=read(out/'container_identity.json');launch=read(out/'launch.json');break
  except (FileNotFoundError,json.JSONDecodeError):
   if time.monotonic()>deadline:raise TimeoutError('container identity')
   time.sleep(.2)
 cid=identity.get('container_id');command=launch['command']
 if not isinstance(cid,str) or len(cid)!=64 or socket.gethostname()!=cid[:12] or cid not in Path('/proc/self/cgroup').read_text() or '--gpus' in command or '--device' in command or launch.get('approval')!=approval:raise ValueError('container identity')
 sys.path.insert(0,str(TIMING));import reference_inputs,module_load
 refs=reference_inputs.load();author=PROJECT/'_reproduction/author_eadp/Qwen_vl';sys.path[:0]=[str(author),str(author/'VLMEvalKit'),str(ROOT/'e6/rres_k256_pilot_v2'),str(ROOT/'e4')]
 import torch,identity_core,worker,decode_observer,resolution,timing_imports,vlmeval.vlm.qwen3_vl.model_fixed_res as fixed
 request,core,timing_runner=timing_imports.load()
 capture=module_load.load('_identity_preflight_capture',ROOT/'e6/rres_k256_pilot_v2/capture.py');adapter=module_load.load('_identity_preflight_adapter',ROOT/'e4_c_budget_pilot_v2/adapter.py')
 paths={'capture':Path(inspect.getfile(capture.Capture)).resolve(),'adapter':Path(inspect.getfile(adapter.BudgetLightAdapter)).resolve(),'decode':Path(decode_observer.__file__).resolve(),'resolution':Path(resolution.__file__).resolve(),'fixed':Path(inspect.getfile(fixed.Qwen3VLChatFixedRes)).resolve(),'timing_runner':Path(timing_runner.__file__).resolve(),'qwen_request':Path(request.__file__).resolve(),'process_core':Path(core.__file__).resolve()}
 expected={'capture':ROOT/'e6/rres_k256_pilot_v2/capture.py','adapter':ROOT/'e4_c_budget_pilot_v2/adapter.py','decode':ROOT/'e4/decode_observer.py','resolution':ROOT/'e6/rres_k256_pilot_v2/resolution.py','fixed':author/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py','timing_runner':TIMING/'paired_runner.py','qwen_request':TIMING/'qwen_request.py','process_core':TIMING/'process_core.py'};plan=gate.plan()
 if any(paths[k]!=v.resolve() or gate.sha(paths[k])!=plan['source_sha256'][str(v.resolve())] for k,v in expected.items()) or torch.cuda.is_initialized() or fixed.QWEN3_FIXED_RESOLUTION!=1024:raise ValueError('module identity or CUDA')
 if len(worker.ANCHORS)!=3 or any(k not in refs['rows'] for k in worker.ANCHORS):raise ValueError('identity anchors')
 result={'status':'passed','scope':'actual diagnostic imports and references only','container_id':cid,'module_paths':{k:str(v) for k,v in paths.items()},'anchors':list(worker.ANCHORS),'model_loaded':False,'cuda_initialized':False,'GPU_authorized':False,'GT_opened':False,'existing_predictions_opened':True,'errors':[]}
 with (out/'worker_preflight.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
