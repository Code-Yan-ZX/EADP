"""32-image genuine selector input capture and three-budget comparison."""
import argparse,gc,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent.parent;os.environ['TRITON_CACHE_DIR']=str(HERE/'triton_cache');sys.path[:0]=[str(BASE/'e4_c_microbench_full'),str(BASE/'e4_c_microbench_pilot'),str(BASE/'e4_c_latency')]
from admission import admit,sources
from latency_worker import prepare,load_runtime,save
from real_worker import sha,preflight_one
from input_preflight import tensor_meta
from runner import durable_journal
from pilot_worker import module,archive_cpu,get_reference
from capture_guard import capture_one
from identity_contract import validate as validate_identity
from panel_reference import get_records
from input_contract import bind_rows,compare_real
import inspect
if Path(inspect.getsourcefile(bind_rows)).resolve()!=HERE/'input_contract.py' or Path(inspect.getsourcefile(compare_real)).resolve()!=HERE/'input_contract.py':raise ValueError('real input module source mismatch')
from trials import author_case,bench
from gates import synthetic_eligibility,endpoint_eligible
from cache_contract import probe_cache

def archive(torch,path,obj):
 with path.open('xb') as f:torch.save(archive_cpu(torch,obj),f);f.flush();os.fsync(f.fileno())
 return sha(path)

def synthetic(torch,native,kernel,GraphSelector):
 torch.manual_seed(20261104);device='cuda';near=torch.ones((8,8),device=device,dtype=torch.float32)*.5;near[2,0]=.50000012;near[3,0]=.50000006
 cases=[('ties',torch.ones((8,8),device=device),torch.ones(8,device=device),4),('zeros',torch.zeros((8,8),device=device),torch.zeros(8,device=device),4),('near_ties',near,torch.ones(8,device=device),4),('seeded_random',torch.rand((32,32),device=device),torch.rand(32,device=device),8),('boundary_one',torch.eye(8,device=device),torch.ones(8,device=device),1),('boundary_all',torch.eye(8,device=device),torch.ones(8,device=device),8)];out=[]
 with torch.inference_mode():
  for name,sim,w,k in cases:
   a=native._greed_select_impl(w[None,:],sim[None,:,:],k)[0][0];b=kernel.fused_greedy(sim,w,k)[0];graph=GraphSelector(sim,w,k);g=graph(sim,w);torch.cuda.synchronize();out.append({'case':name,'k':k,'fused_equal':bool(torch.equal(a,b)),'graph_equal':bool(torch.equal(a,g)),'author_ids':a.cpu().tolist(),'fused_ids':b.cpu().tolist(),'graph_ids':g.cpu().tolist()});del graph
 # Confirm that the same captured graph actually consumes changed numeric inputs.
 with torch.inference_mode():
  s=torch.ones((8,8),device='cuda');w=torch.ones(8,device='cuda');g=GraphSelector(s,w,4);before=g(s,w).clone();changed=torch.zeros_like(s);changed[-1,:]=1;expected=native._greed_select_impl(w[None,:],changed[None,:,:],4)[0][0];after=g(changed,w);torch.cuda.synchronize();out.append({'case':'graph_changed_input','k':4,'graph_equal':bool(torch.equal(after,expected)),'different_from_original':not bool(torch.equal(before,after)),'before_graph_ids':before.cpu().tolist(),'author_ids':expected.cpu().tolist(),'graph_ids':after.cpu().tolist()});del g
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args();out=a.out.resolve();approval=admit(preflight_only=a.preflight_only,host=False);launch=json.loads((out/'launch.json').read_text());source={str(p):sha(p) for p in sources()}
 if launch.get('approval_bundle')!=approval or launch.get('worker_sha256')!=sha(__file__) or source!=approval['source_sha256']:raise ValueError('launch/source mismatch')
 cache=probe_cache(HERE/'triton_cache')
 if os.environ.get('TRITON_CACHE_DIR')!=str(HERE/'triton_cache'):raise ValueError('Triton cache env mismatch')
 manifest=json.loads((HERE/'sample_manifest_001.json').read_text());frozen=json.loads((BASE/'e4_c_microbench/schedule001.json').read_text());prepared=prepare('smoke');rows=bind_rows(frozen,prepared)
 if len(rows)!=32 or len(manifest['full_128'])!=128 or manifest['budgets']!=[128,256,512] or manifest['pilot']!=[{k:r[k] for k in ('task','qid','group_id','message_sha256','image_sha256')} for r in frozen['input_identities']]:raise ValueError('frozen selector sample mismatch')
 records=get_records(manifest['pilot'],approval['prior_identity']['e4']);torch,native,wrapper_class,versions=load_runtime(prepared);capture=module(BASE/'e4_c_microbench/capture.py','real_selector_capture');light=module(BASE/'e4_c_latency/light_adapter.py','real_selector_light');kernel=module(BASE/'e5/fused_selector/kernel.py','real_selector_fused')
 if 'kernel' in sys.modules:raise ValueError('preloaded kernel module')
 sys.modules['kernel']=kernel;graph_module=module(BASE/'e5/graph_selector/graph.py','real_selector_graph');from vlmeval.vlm.qwen3_vl.model_fixed_res import Qwen3VLChatCDPruner
 if torch.cuda.is_initialized():raise ValueError('CUDA initialized before model construction')
 if a.preflight_only:
  save(out/'worker_preflight.json',{'status':'passed','images':32,'cases':96,'e4_identity_records':len(records),'cuda_initialized':False,'model_loaded':False,'predictions_opened':True,'prediction_scope':'accepted E1 and E4 DEV mixed records parsed only for input and selector identity','GT_opened':False,'scoring_performed':False,'source_sha256':source,'weight_stats_before':prepared['weights_before'],'versions':versions,'triton_cache':cache});return
 if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
 start=time.monotonic();wrapper=wrapper_class(**prepared['plan']['recipe']);load_seconds=time.monotonic()-start
 if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('model backend mismatch')
 refs=[get_reference(row) for row in rows];device=torch.device('cuda:0');handle,journal_raw=durable_journal(out/'journal.jsonl');counts={'cases':0,'images':0};syn=None;graphs={}
 try:
  syn=synthetic(torch,native,kernel,graph_module.GraphSelector)
  try:allowed=synthetic_eligibility(syn);syn_error=None
  except ValueError as exc:allowed={'fused':False,'graph':False};syn_error=str(exc)
  journal_raw({'status':'synthetic_completed','cases':syn,'eligibility':allowed,'validation_error':syn_error})
  for i,(row,ref) in enumerate(zip(rows,refs)):
   def journal(event):journal_raw(dict(event,image_index=i))
   journal({'status':'image_started','task':row['task'],'qid':str(row['qid']),'group_id':row['group_id']});image_dir=out/('image_%03d'%i);image_dir.mkdir();holder={}
   class Capture(capture.CaptureInputs):
    def __init__(self,*args):super().__init__(*args);holder['instance']=self
   try:
    preflight_one(torch,wrapper,row,ref);ids=wrapper._tokenize_instruction(row['message'],dataset=row['task'])[0].detach().cpu().tolist()
    ops,base,actual,clean=capture_one(torch,wrapper,native,row,Qwen3VLChatCDPruner.generate_inner_transformers_fixed,light.LightAdapter,Capture,capture.clone_tree,journal)
    if not clean():raise RuntimeError('hook cleanup failed')
    state=holder['instance'].state['selector'];args=state['args'];kw=state['kwargs']
    if len(args)!=4 or kw:raise ValueError('unexpected actual selector signature')
    try:identity=validate_identity(actual,ref,tensor_meta)
    except ValueError as exc:identity={'status':'failed','reason':str(exc)}
    try:reconstruction=compare_real(torch,actual,args,ids,row['image_start']);reconstruction['status']='passed' if reconstruction['sequence_equal'] and reconstruction['mean_equal'] else 'mismatch'
    except ValueError as exc:reconstruction={'status':'failed','reason':str(exc)}
    visual_arg=bool(torch.equal(args[0],actual['visual'][0]));digest=archive(torch,image_dir/'capture.pt',{'actual':actual,'selector_args':args,'selector_output':base['selector'],'instruction_ids':ids});journal({'status':'capture_archived','task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'sha256':digest,'identity':identity,'reconstruction':reconstruction,'visual_arg_equal':visual_arg})
   except ValueError as exc:
    journal({'status':'capture_failed','task':row['task'],'qid':str(row['qid']),'group_id':row['group_id'],'reason':str(exc)})
    for k in (128,256,512):journal({'status':'case_completed','budget':k,'result':'not_evaluated','reason':'capture_failed'});counts['cases']+=1
    journal({'status':'image_completed','task':row['task'],'qid':str(row['qid']),'group_id':row['group_id']});counts['images']+=1;continue
   meta={'processor':{k:tensor_meta(actual['processor'][k]) for k in ('input_ids','attention_mask','pixel_values','image_grid_thw')},'main_visual':tensor_meta(actual['visual'][0]),'full_deepstack':[tensor_meta(x) for x in actual['visual'][1]]};numeric={}
   for k in (128,256,512):
    record=records[(row['task'],str(row['qid']),k)]
    try:
     result,importance,similarity,ordered,pruned=author_case(torch,native,args,k,meta,record,tensor_meta);case_digest=archive(torch,image_dir/('case_%d.pt'%k),{'importance':importance,'similarity':similarity,'ordered':ordered,'pruned':pruned});numeric[k]=(result,importance,similarity,ordered);journal({'status':'author_case_archived','budget':k,'sha256':case_digest,'result':result})
    except ValueError as exc:numeric[k]={'reason':str(exc)};journal({'status':'author_case_failed','budget':k,'reason':str(exc)})
   baseline=numeric.get(256);cross={}
   for k in (128,256,512):
    case=numeric[k]
    if isinstance(case,dict) or isinstance(baseline,dict):cross[str(k)]={'available':False}
    else:cross[str(k)]={'available':True,'importance_equal_256':bool(torch.equal(case[1],baseline[1])),'similarity_equal_256':bool(torch.equal(case[2],baseline[2]))}
   journal({'status':'cross_budget_comparison','values':cross});capture_order_equal=not isinstance(baseline,dict) and bool(torch.equal(baseline[3].reshape(-1),actual['ordered'].reshape(-1)))
   for k in (128,256,512):
    case=numeric[k]
    if isinstance(case,dict):result={'status':'not_evaluated','reason':case['reason']}
    else:
     author,importance,similarity,ordered=case
     if not endpoint_eligible(identity,reconstruction,visual_arg,author,cross[str(k)],capture_order_equal):
      result={'status':'not_evaluated','reason':'author_or_input_identity_gate_failed','author_endpoint':author,'cross_budget':cross[str(k)],'capture_order_equal':capture_order_equal}
     else:
      try:
       preliminary,measurements,setup=bench(torch,native,kernel,graph_module.GraphSelector,graphs,similarity,importance,k,ordered[0],journal,allowed)
       if preliminary['author_equals_endpoint'] is not True:result={'status':'not_evaluated','reason':'author_repeat_identity_failed','identity':preliminary,'author_endpoint':author,'cross_budget':cross[str(k)]}
       else:
        author_stable=all(m.get('equals_author') is True for m in measurements if m['implementation']=='author');matches={name:author_stable and bool(preliminary.get(name+'_equals_author')) and all(m.get('equals_author') is True for m in measurements if m['implementation']==name) for name in ('fused','graph')}
        result={'status':'measured','candidate_matches_author':matches,'identity':preliminary,'measurements':measurements,'author_endpoint':author,'cross_budget':cross[str(k)]}
      except ValueError as exc:result={'status':'not_evaluated','reason':str(exc),'author_endpoint':author,'cross_budget':cross[str(k)]}
    journal({'status':'case_completed','budget':k,'result':result});counts['cases']+=1
   journal({'status':'image_completed','task':row['task'],'qid':str(row['qid']),'group_id':row['group_id']});counts['images']+=1;del actual,args,holder,numeric,base,ops;gc.collect();torch.cuda.synchronize(device)
 finally:handle.close()
 if counts!={'cases':96,'images':32} or any(sha(p)!=h for p,h in source.items()):raise ValueError('incomplete cases or changed source')
 for path,before in prepared['weights_before'].items():
  s=Path(path).stat()
  if [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]!=before:raise ValueError('weight identity changed')
 save(out/'worker_complete.json',{'status':'completed','images':32,'cases':96,'diagnostic_synthetic':syn,'synthetic_validation_error':syn_error,'source_sha256':source,'model_load_seconds':load_seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'versions':versions,'triton_cache':cache,'predictions_opened':True,'GT_opened':False,'scoring_performed':False,'scope':'all planned cases retained; completed does not imply reconstruction or fused identity passed'})
if __name__=='__main__':main()
