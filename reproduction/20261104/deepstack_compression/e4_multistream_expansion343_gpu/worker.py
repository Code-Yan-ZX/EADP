"""Real Qwen3 nine-image pilot entry: CPU preflight; GPU remains locked."""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;P=B/'e4_multistream_expansion352'
# File identity is checked before executing the admission module.
ADMISSION_SHA='ec745703df0789820f7ca64bcaa89f6b37eaae1df21768c5a5c7077746b2e765'
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
   out[key]={k:r[k] for k in ('task','qid','group_id','image_sha256','message_sha256','budget','ds_on','selection_mode','actual_injection_verified','processor','image_start','main_visual','full_deepstack','ordered_ids','spatial_ids','prefill_embeds','prefill_mask','prefill_position_ids','adjusted_delta','injected_mask','injected_deepstack')}
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
 cohort=load('_expansion343_cohort',D/'cohort.py',approval['source_sha256'][str((D/'cohort.py').resolve())]);selected=cohort.build(adm.read(P/'selection001.json'));prepared=latency_worker.prepare('smoke');refs=panel_reference.accepted();records=references(selected,refs)
 if len(selected)!=343 or len(records)!=343:raise ValueError('sample coverage')
 # prepare() has already validated the complete E1 proof and journal SHA.
 # Its refs are deliberately slim P3-D0; full E1 P3-D1 is required below.
 fr=load('_expansion352v4_fullref',D/'full_reference.py',approval['source_sha256'][str((D/'full_reference.py').resolve())]);proof=adm.read(B/'e1/e1full001_completion.json');entry=proof['files']['journal'];path=Path(entry['path'])
 if adm.sha(path)!=entry['sha256']:raise ValueError('E1 journal changed')
 with path.open() as f:full_refs=fr.collect(selected,(json.loads(x) for x in f if x.strip()),prepared['refs'],records)
 if adm.sha(path)!=entry['sha256']:raise ValueError('E1 journal changed during read')
 rows={(r['task'],str(r['qid'])):r for r in prepared['rows']}
 if len(rows)!=384 or any((r['task'],str(r['qid'])) not in rows or any(str(rows[(r['task'],str(r['qid']))].get(k))!=str(r.get(k)) for k in ('task','qid','group_id','image_sha256','message_sha256')) for r in selected):raise ValueError('prepared E1 inputs')
 torch,native,wrapper_class,versions=latency_worker.load_runtime(prepared)
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatCDPruner
 cap=load('_expansion352_capture',B/'e4_c_microbench/capture.py',approval['source_sha256'][str((B/'e4_c_microbench/capture.py').resolve())]);light=load('_expansion352_light',B/'e4_c_latency/light_adapter.py',approval['source_sha256'][str((B/'e4_c_latency/light_adapter.py').resolve())])
 if torch.cuda.is_initialized():raise ValueError('CUDA initialized before model')
 if a.preflight_only:
  from types import SimpleNamespace
  import inspect
  from transformers import AutoProcessor,AutoConfig
  from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatFixedRes
  author=Path(prepared['plan']['author_root'])/'VLMEvalKit/vlmeval/vlm/qwen3_vl/model_fixed_res.py'
  if Path(inspect.getfile(Qwen3VLChatFixedRes)).resolve()!=author.resolve():raise ValueError('author processor wrapper source')
  model=Path('/home/dell/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
  processor=AutoProcessor.from_pretrained(model,local_files_only=True,trust_remote_code=False);config=AutoConfig.from_pretrained(model,local_files_only=True,trust_remote_code=False)
  w=object.__new__(Qwen3VLChatFixedRes);w.system_prompt=None;w.min_pixels=w.max_pixels=1048576;w.total_pixels=None;w.processor=processor;w.model=SimpleNamespace(device='cpu',dtype=torch.bfloat16)
  pp=load('_expansion352v2_processor',D/'processor_preflight.py',approval['source_sha256'][str((D/'processor_preflight.py').resolve())]);checked=pp.check(torch,w,config,selected,prepared['rows'],prepared['refs'],records,input_preflight.validate_processor)
  latency_worker.save(out/'worker_preflight.json',{'status':'passed','images':343,'processor_inputs':checked,'processor_inputs_checked':len(checked),'full_inputs':384,'e4_reference_records':len(records),'e1_p3d1_complete_references':len(full_refs),'source_sha256':approval['source_sha256'],'selection_sha256':approval['selection_sha256'],'weight_stats_before':prepared['weights_before'],'versions':versions,'model_loaded':False,'cuda_initialized':bool(torch.cuda.is_initialized()),'GT_opened':False,'predictions_opened':True,'prediction_scope':'accepted E1/E4 DEV mixed journals parsed only for matching identity; no scoring or new reference labels','scoring_performed':False});return 0
 # GPU path only after separately reviewed v2 CPU and actual container preflight.
 from runner import durable_journal
 engine=load('_expansion352_engine',D/'pilot_engine.py',approval['source_sha256'][str((D/'pilot_engine.py').resolve())]);norm=load('_expansion352_norm',P/'norm_contract.py',approval['source_sha256'][str((P/'norm_contract.py').resolve())]);contract=load('_expansion352_contract',P/'pilot_contract.py',approval['source_sha256'][str((P/'pilot_contract.py').resolve())]);features=norm.original()
 if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
 start=time.monotonic();wrapper=wrapper_class(**prepared['plan']['recipe']);model_load=time.monotonic()-start
 if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('backend')
 handle,journal=durable_journal(out/'journal.jsonl');requests_started=time.monotonic()
 try:
  journal({'status':'model_loaded','model_load_seconds':model_load});engine.run(selected,prepared['rows'],full_refs,records,out=out/'captures',torch=torch,wrapper=wrapper,native=native,author_generate=Qwen3VLChatCDPruner.generate_inner_transformers_fixed,Adapter=light.LightAdapter,Capture=cap.CaptureInputs,clone=cap.clone_tree,capture_one=capture_guard.capture_one,preflight_one=real_worker.preflight_one,validate_e1=identity_contract.validate,tensor_meta=input_preflight.tensor_meta,identity_check=contract.check_identity,norm_mod=norm,feature_mod=features,journal=journal)
 finally:handle.close()
 requests_seconds=time.monotonic()-requests_started;peak_allocated=torch.cuda.max_memory_allocated();peak_reserved=torch.cuda.max_memory_reserved();cleanup_started=time.monotonic()
 import gc
 del wrapper;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();cleanup_seconds=time.monotonic()-cleanup_started
 after={}
 for path,before in prepared['weights_before'].items():
  st=Path(path).stat();after[path]=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
 if after!=prepared['weights_before'] or any(adm.sha(k)!=v for k,v in approval['source_sha256'].items()):raise ValueError('weight or source identity changed')
 latency_worker.save(out/'worker_complete.json',{'status':'completed','images':343,'model_load_seconds':model_load,'peak_allocated_bytes':peak_allocated,'peak_reserved_bytes':peak_reserved,'requests_seconds':requests_seconds,'cleanup_seconds':cleanup_seconds,'weight_stats_before':prepared['weights_before'],'weight_stats_after':after,'source_sha256':approval['source_sha256'],'versions':versions,'GT_opened':False,'scoring_performed':False});return 0
if __name__=='__main__':raise SystemExit(main())
