"""Candidate same-image worker. Admission must pass before processor/model use."""
import argparse,hashlib,importlib.util,json,os,sys,time,traceback
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parent;LAT=B/'e4_c_latency';E4=B/'e4';RS=B/'e6/random_smoke'

def load(name,path):
 path=Path(path).resolve()
 if name in sys.modules:raise ValueError('preloaded module '+name)
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m
 try:spec.loader.exec_module(m)
 except BaseException:del sys.modules[name];raise
 return m

def gate():
 p=D/'admission.py'
 if hashlib.sha256(p.read_bytes()).hexdigest()!='eabc64818fefeefd46563bce12a547b093b7165ca3fa2819c1fb576a2804aef4':raise ValueError('unreviewed gate')
 return load('_minimax32_worker_gate',p)

def save(path,obj):
 with Path(path).open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())

def check_cgroup(path):
 deadline=time.monotonic()+30
 while True:
  try:identity=json.loads(Path(path).read_text());break
  except (FileNotFoundError,json.JSONDecodeError):
   if time.monotonic()>=deadline:raise ValueError('container identity unavailable')
   time.sleep(.1)
 cid=identity.get('container_id');own=Path('/proc/self/cgroup').read_text()
 if not isinstance(cid,str) or len(cid)!=64 or cid not in own or cid not in identity.get('host_cgroup','') or not (identity.get('host_init') or {}).get('start_ticks'):raise ValueError('container identity mismatch')
 return identity

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--preflight-only',action='store_true');a=parser.parse_args();out=a.out.resolve()
 if not out.is_dir():raise ValueError('launcher output missing')
 g=gate();approval=g.admit(a.preflight_only);plan=g.read(D/'execution_plan001.json');launch=g.read(out/'launch.json')
 if launch.get('approval_bundle')!=approval or launch.get('worker_sha256')!=g.sha(__file__) or launch.get('run_out')!=str(out):raise ValueError('launch/source mismatch')
 check_cgroup(out/'container_identity.json')
 sys.path[:0]=[str(D),str(E4),str(LAT),str(RS),str(B/'e4_c_microbench_pilot')]
 controlled={**{n:D/(n+'.py') for n in ('cohort','arm','panel')},**{n:E4/(n+'.py') for n in ('budget_adapter','decode_observer','identity_records','off_adapter','skip_deepstack')},**{n:LAT/(n+'.py') for n in ('latency_gate','real_worker','runner','stage_runner','schedule','contract','timer','input_preflight','e1_reference_gate','identity_gate','light_adapter','memory_probe')}}
 for name in controlled:
  if name in sys.modules:raise ValueError('unexpected preloaded module '+name)
 latency=load('_minimax32_latency',LAT/'latency_worker.py');e1=load('_minimax32_e1',B/'m3_oracle_exec/e1_admission.py');refs=load('_minimax32_refs',B/'e6/random_full_exec_v2/references.py')
 selected=load('_minimax32_selected',B/'m3_oracle_exec/selection_adapter.py');runner=load('_minimax32_budget_runner',E4/'budget_runner.py');identity=load('_minimax32_identity',RS/'identity_contract.py')
 import cohort,arm,panel
 for name,path in controlled.items():
  m=sys.modules.get(name)
  if m is not None and Path(getattr(m,'__file__','')).resolve()!=path.resolve():raise ValueError('actual module path '+name)
 prepared=latency.prepare('smoke');receipt=e1.validate(metadata_only=False);reference_lists=refs.load(B/'e1/e1full001/raw_predictions.jsonl',receipt,prepared['rows'])
 index={(r['task'],str(r['qid'])):r for r in prepared['rows']};items=g.read(D/'cohort001.json')['cohort'];rows=[];references={}
 if len(items)!=32 or len(index)!=384:raise ValueError('cohort size')
 for item in items:
  key=(item['task'],str(item['qid']));row=index[key];ref=reference_lists[key][0]
  if any(str(row.get(k))!=str(item[k]) for k in ('task','qid','group_id','image_sha256','message_sha256')):raise ValueError('cohort input binding')
  if ref['processor']!=item['processor']:raise ValueError('P3 reference processor mismatch')
  rows.append(row);references[key]=ref
 torch,native,wrapper_class,versions=latency.load_runtime(prepared)
 from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatCDPruner,Qwen3VLChatFixedRes
 if a.preflight_only:
  if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('CPU preflight exposed CUDA')
  from transformers import AutoProcessor,AutoConfig
  from types import SimpleNamespace
  recipe=prepared['plan']['recipe'];processor=AutoProcessor.from_pretrained(recipe['model_path'],local_files_only=True,trust_remote_code=False);config=AutoConfig.from_pretrained(recipe['model_path'],local_files_only=True,trust_remote_code=False)
  w=object.__new__(Qwen3VLChatFixedRes);w.system_prompt=None;w.min_pixels=w.max_pixels=1048576;w.total_pixels=None;w.fps=None;w.processor=processor;w.model=SimpleNamespace(device='cpu',dtype=torch.bfloat16,config=config)
  checked=[]
  for row in rows:
   key=(row['task'],str(row['qid']));latency.preflight_one(torch,w,row,prepared['refs'][key]);checked.append({'task':row['task'],'qid':str(row['qid']),'processor':prepared['refs'][key]['processor']})
  if len(checked)!=32 or torch.cuda.is_initialized():raise ValueError('processor/CUDA preflight')
  save(out/'worker_preflight.json',{'status':'passed','questions':32,'generations_planned':64,'processor_checked':checked,'model_loaded':False,'cuda_initialized':False,'GT_opened':False,'scoring_performed':False,'existing_DEV_predictions_opened':True,'prediction_scope':'accepted E1 DEV mixed journal parsed only for identities','weights_before':prepared['weights_before'],'versions':versions,'errors':[]});return
 if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('GPU identity')
 torch.cuda.set_per_process_memory_fraction(.9,0)
 panel.event(out/'heartbeat.jsonl',{'event':'model_loading','unix':time.time()})
 wrapper=wrapper_class(**prepared['plan']['recipe'])
 if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('backend mismatch')
 from capture_guard import hook_inventory,hooks_equal,instance_attr,restored
 counter={'steps':0}
 def heartbeat(_m,_args,_kwargs,_output):
  counter['steps']+=1
  if counter['steps']%32==0:panel.event(out/'heartbeat.jsonl',{'event':'forward','steps':counter['steps'],'unix':time.time()})
 heartbeat_hook=wrapper.model.model.language_model.register_forward_hook(heartbeat,with_kwargs=True)
 def process(row,ref):return latency.preflight_one(torch,wrapper,row,prepared['refs'][(row['task'],str(row['qid']))])
 def run_one(row,name,ids,ref,on_generated,on_raw):
  modules=list(wrapper.model.modules())+list(wrapper.pruner.modules());seen=set();modules=[m for m in modules if not (id(m) in seen or seen.add(id(m)))];hooks=hook_inventory(modules);greedy=native._greed_select_impl;before_generate=instance_attr(wrapper.model,'generate');before_processor=instance_attr(wrapper,'_processor_inputs')
  def generate(w,r):
   with torch.inference_mode():return Qwen3VLChatCDPruner.generate_inner_transformers_fixed(w,r['message'],dataset=r['task'])
  def archive(tensors):return {'kind':'summary_only','tensors':{k:sys.modules['identity_records'].tensor_meta(v) for k,v in tensors.items()}}
  try:
   result=arm.run_arm(torch,wrapper,native,row,name,ids,ref,runner,selected.SelectedAdapter,generate,on_generated,archive,sys.modules['identity_records'].tensor_meta,on_raw)
   identity.validate_e1_identity(result,row,ref)
  finally:
   clean=hooks_equal(hooks) and native._greed_select_impl is greedy and restored(wrapper.model,'generate',before_generate) and restored(wrapper,'_processor_inputs',before_processor)
   if not clean:raise RuntimeError('hook/patch cleanup failure')
  return {**result,'cleanup_passed':True}
 def complete():
  heartbeat_hook.remove()
  if heartbeat_hook.id in wrapper.model.model.language_model._forward_hooks:raise ValueError('heartbeat hook not removed')
  after={}
  for path,before in prepared['weights_before'].items():
   st=Path(path).stat();after[path]=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
   if after[path]!=before:raise ValueError('weight identity changed')
  if any(g.sha(k)!=v for k,v in plan['source_sha256'].items()):raise ValueError('source changed')
  save(out/'completion_provenance.json',{'status':'passed','weights_before':prepared['weights_before'],'weights_after':after,'source_sha256':plan['source_sha256'],'GT_opened':False,'scoring_performed':False})
 try:panel.run(items,rows,references,out,process,run_one,complete)
 finally:heartbeat_hook.remove()

if __name__=='__main__':
 try:main()
 except BaseException as exc:
  try:
   args=sys.argv
   if args.count('--out')==1:
    out=Path(args[args.index('--out')+1])
    if out.is_dir() and not (out/'failure.json').exists():save(out/'failure.json',{'status':'failed','error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'GT_opened':False})
  finally:raise
