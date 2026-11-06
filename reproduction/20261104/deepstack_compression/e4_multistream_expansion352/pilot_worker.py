"""Real Qwen3 nine-image pilot entry: CPU preflight; GPU remains locked."""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent
# File identity is checked before executing the admission module.
ADMISSION_SHA='483095eee79d9ad74312e5b70fb3cf6ec3c3f3525e97351c886721927f7e35e3'
def load(name,path,expected):
 p=Path(path).resolve()
 if hashlib.sha256(p.read_bytes()).hexdigest()!=expected or name in sys.modules:raise ValueError('preloaded or changed '+name)
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m
 try:s.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m
def references(rows,identity):
 path=B/'e4_panel_v2/panel_003/records.jsonl'
 from identity_gate import sha
 if sha(path)!=identity['records_sha256']:raise ValueError('E4 source changed')
 wanted={(r['task'],str(r['qid'])):r for r in rows};out={}
 with path.open() as f:
  for line in f:
   r=json.loads(line);key=(r.get('task'),str(r.get('qid')))
   if key not in wanted or r.get('budget')!=256 or r.get('ds_on') is not True:continue
   if key in out or r.get('selection_mode')!='author_greedy' or r.get('actual_injection_verified') is not True or any(str(r.get(k))!=str(wanted[key].get(k)) for k in ('task','qid','group_id','image_sha256','message_sha256')):raise ValueError('E4 record identity')
   out[key]={k:r[k] for k in ('task','qid','group_id','image_sha256','message_sha256','budget','ds_on','selection_mode','actual_injection_verified','processor','main_visual','full_deepstack')}
 if len(out)!=len(wanted) or sha(path)!=identity['records_sha256']:raise ValueError('E4 record coverage')
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args();out=a.out.resolve()
 adm=load('_expansion352_admission',D/'admission.py',ADMISSION_SHA);approval=adm.admit(a.preflight_only)
 launch=adm.read(out/'launch.json')
 if launch.get('approval_bundle')!=approval or launch.get('worker_sha256')!=adm.sha(__file__):raise ValueError('launch identity')
 # Reject colliding names before the real import chain is executed.
 for name in ('latency_worker','real_worker','input_preflight','capture_guard','identity_contract','panel_reference','identity_gate','latency_gate','runner'):
  if name in sys.modules:raise ValueError('preloaded dependency '+name)
 # Match the previously independently reviewed real capture chain and verify actual paths.
 sys.path[:0]=[str(B/'e4_c_microbench_full'),str(B/'e4_c_microbench_pilot'),str(B/'e4_c_latency'),str(B/'e5/selector_real32')]
 import latency_worker,real_worker,input_preflight,capture_guard,identity_contract,panel_reference
 expected=[(latency_worker,B/'e4_c_latency/latency_worker.py'),(real_worker,B/'e4_c_latency/real_worker.py'),(input_preflight,B/'e4_c_latency/input_preflight.py'),(capture_guard,B/'e4_c_microbench_pilot/capture_guard.py'),(identity_contract,B/'e4_c_microbench_pilot/identity_contract.py'),(panel_reference,B/'e5/selector_real32/panel_reference.py')]
 if any(Path(m.__file__).resolve()!=p.resolve() for m,p in expected):raise ValueError('actual import source')
 selected=adm.read(D/'selection001.json')['pilot'];prepared=latency_worker.prepare('smoke');refs=panel_reference.accepted();records=references(selected,refs)
 if len(selected)!=9 or len(records)!=9:raise ValueError('sample coverage')
 rows={(r['task'],str(r['qid'])):r for r in prepared['rows']}
 if len(rows)!=384 or any((r['task'],str(r['qid'])) not in rows or any(str(rows[(r['task'],str(r['qid']))].get(k))!=str(r.get(k)) for k in ('task','qid','group_id','image_sha256','message_sha256')) for r in selected):raise ValueError('prepared E1 inputs')
 torch,native,wrapper_class,versions=latency_worker.load_runtime(prepared)
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatCDPruner
 cap=load('_expansion352_capture',B/'e4_c_microbench/capture.py',approval['source_sha256'][str((B/'e4_c_microbench/capture.py').resolve())]);light=load('_expansion352_light',B/'e4_c_latency/light_adapter.py',approval['source_sha256'][str((B/'e4_c_latency/light_adapter.py').resolve())])
 if torch.cuda.is_initialized():raise ValueError('CUDA initialized before model')
 if a.preflight_only:
  latency_worker.save(out/'worker_preflight.json',{'status':'passed','pilot_images':9,'full_inputs':384,'e4_reference_records':len(records),'source_sha256':approval['source_sha256'],'selection_sha256':approval['selection_sha256'],'weight_stats_before':prepared['weights_before'],'versions':versions,'model_loaded':False,'cuda_initialized':False,'GT_opened':False,'predictions_opened':True,'prediction_scope':'accepted E1/E4 DEV mixed journals parsed only for matching identity; no scoring or new reference labels','scoring_performed':False});return 0
 # Unreachable while admit(False) is explicitly locked; left as reviewed real implementation.
 from runner import durable_journal
 engine=load('_expansion352_engine',D/'pilot_engine.py',approval['source_sha256'][str((D/'pilot_engine.py').resolve())]);norm=load('_expansion352_norm',D/'norm_contract.py',approval['source_sha256'][str((D/'norm_contract.py').resolve())]);contract=load('_expansion352_contract',D/'pilot_contract.py',approval['source_sha256'][str((D/'pilot_contract.py').resolve())]);features=norm.original()
 if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
 start=time.monotonic();wrapper=wrapper_class(**prepared['plan']['recipe']);model_load=time.monotonic()-start
 if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('backend')
 handle,journal=durable_journal(out/'journal.jsonl')
 try:
  journal({'status':'model_loaded','model_load_seconds':model_load});engine.run(selected,prepared['rows'],prepared['refs'],records,out=out/'captures',torch=torch,wrapper=wrapper,native=native,author_generate=Qwen3VLChatCDPruner.generate_inner_transformers_fixed,Adapter=light.LightAdapter,Capture=cap.CaptureInputs,clone=cap.clone_tree,capture_one=capture_guard.capture_one,preflight_one=real_worker.preflight_one,validate_e1=identity_contract.validate,tensor_meta=input_preflight.tensor_meta,identity_check=contract.check_identity,norm_mod=norm,feature_mod=features,journal=journal)
 finally:handle.close()
 latency_worker.save(out/'worker_complete.json',{'status':'completed','images':9,'model_load_seconds':model_load,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'GT_opened':False,'scoring_performed':False});return 0
if __name__=='__main__':raise SystemExit(main())
