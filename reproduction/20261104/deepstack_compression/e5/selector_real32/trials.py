"""Real author per-budget forward and candidate identity, with explicit failures."""
import math,time

def author_case(torch,native,args,k,meta,record,tensor_meta):
 if len(args)!=4 or k not in (128,256,512):raise ValueError('bad author arguments or K')
 pruner=native.VisualTokenPruner(visual_token_num=k,alpha=.5,beta=2.,visual_dim=4096,spatial_merge_size=2,device=str(args[0].device)).to(args[0].device).eval()
 if (pruner.visual_token_num,pruner.alpha,pruner.beta,pruner.visual_dim,pruner.spatial_merge_size)!=(k,.5,2.,4096,2):raise ValueError('author pruner config mismatch')
 saved=native._greed_select_impl;capture=[]
 def observed(importance,similarity,budget):
  result=saved(importance,similarity,budget);capture.append((importance.detach().clone(),similarity.detach().clone(),result[0].detach().clone()));return result
 native._greed_select_impl=observed
 try:
  with torch.inference_mode():out=pruner(*args)
 finally:native._greed_select_impl=saved
 if native._greed_select_impl is not saved or len(capture)!=1:raise ValueError('greedy observation/cleanup mismatch')
 importance,similarity,ordered=capture[0];ids=ordered[0].detach().cpu().tolist();actual_sorted=sorted(ids)
 common=meta['processor']==record['processor'] and meta['main_visual']==record['main_visual'] and meta['full_deepstack']==record['full_deepstack']
 result={'budget':k,'author_config':{'visual_token_num':k,'alpha':.5,'beta':2.,'visual_dim':4096,'spatial_merge_size':2},'e4_common_metadata_equal':common,'e4_order_equal':ids==record['ordered_ids'],'e4_spatial_equal':actual_sorted==record['spatial_ids'],'e4_pruned_hash_equal':tensor_meta(out[0])==record['pruned_visual'],'ordered_ids':ids,'spatial_ids':actual_sorted,'pruned_meta':tensor_meta(out[0]),'similarity_meta':tensor_meta(similarity),'importance_meta':tensor_meta(importance)}
 return result,importance,similarity,ordered,out

def numeric_preconditions(torch,similarity,importance,k):
 if similarity.shape!=(1,1024,1024) or importance.shape!=(1,1024) or similarity.dtype!=torch.float32 or importance.dtype!=torch.float32 or not similarity.is_contiguous() or not importance.is_contiguous() or k not in (128,256,512):raise ValueError('unsupported fused input layout/dtype')
 if not bool(torch.isfinite(similarity).all()) or not bool(torch.isfinite(importance).all()) or similarity.min().item()<0 or similarity.max().item()>1 or importance.min().item()<0:raise ValueError('unsupported fused input values')

def measurement_order(repeat):
 return [('author','fused','graph'),('fused','graph','author'),('graph','author','fused'),('author','graph','fused'),('fused','author','graph')][repeat]

def bench(torch,native,kernel,GraphSelector,graphs,similarity,importance,k,expected,journal,synthetic_allowed):
 numeric_preconditions(torch,similarity,importance,k)
 def original():return native._greed_select_impl(importance,similarity,k)[0][0]
 def candidate():return kernel.fused_greedy(similarity[0],importance[0],k)[0]
 with torch.inference_mode():
  first=original();torch.cuda.synchronize()
  if not bool(torch.equal(first,expected)):return {'author_equals_endpoint':False},[],None
  second=candidate() if synthetic_allowed['fused'] else None;torch.cuda.synchronize();setup=None
  if synthetic_allowed['graph'] and k not in graphs:
   before={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()};t0=time.perf_counter_ns()
   try:
    graphs[k]=GraphSelector(similarity[0],importance[0],k);torch.cuda.synchronize();setup={'budget':k,'status':'passed','capture_wall_ms':(time.perf_counter_ns()-t0)/1e6,'before_bytes':before,'after_bytes':{'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()},'input_shape':list(similarity[0].shape),'input_copied_each_replay':True}
   except (ValueError,RuntimeError) as exc:
    if isinstance(exc,torch.cuda.OutOfMemoryError):raise
    torch.cuda.synchronize();graphs[k]={'failure':str(exc)};setup={'budget':k,'status':'failed','reason':str(exc),'capture_wall_ms':(time.perf_counter_ns()-t0)/1e6,'before_bytes':before,'after_bytes':{'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}}
   journal({'status':'graph_setup','budget':k,'result':setup})
  graph=graphs.get(k,{'failure':'synthetic_identity_failed'})
  def graphed():return graph(similarity[0],importance[0])
  third=graphed() if not isinstance(graph,dict) else None;torch.cuda.synchronize();initial={'author_equals_endpoint':True,'fused_equals_author':None if second is None else bool(torch.equal(second,first)),'graph_equals_author':None if third is None else bool(torch.equal(third,first)),'fused_ordered_ids':None if second is None else second.detach().cpu().tolist(),'graph_ordered_ids':None if third is None else third.detach().cpu().tolist()};measurements=[];methods={'author':original,'fused':candidate,'graph':graphed};allowed={'author':True,'fused':initial['fused_equals_author'] is True,'graph':initial['graph_equals_author'] is True}
  for repeat in range(5):
   for label in measurement_order(repeat):
    if not allowed[label]:
     m={'repeat':repeat,'implementation':label,'status':'unavailable','reason':'identity_or_synthetic_gate_failed'};measurements.append(m);journal({'status':'case_measurement','budget':k,'measurement':m});continue
    torch.cuda.synchronize();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start=time.perf_counter_ns();begin.record();selected=methods[label]();end.record();end.synchronize();wall=(time.perf_counter_ns()-start)/1e6;ms=float(begin.elapsed_time(end));same=bool(torch.equal(selected,expected))
    if not math.isfinite(ms) or ms<0 or not math.isfinite(wall) or wall<0:raise ValueError('invalid event time')
    m={'repeat':repeat,'implementation':label,'status':'measured','gpu_event_ms':ms,'host_wall_ms':wall,'equals_author':same,'ordered_ids':selected.detach().cpu().tolist()};measurements.append(m);journal({'status':'case_measurement','budget':k,'measurement':m})
    if not same:
     allowed[label]=False
     if label=='author':allowed['fused']=allowed['graph']=False
 return initial,measurements,setup
