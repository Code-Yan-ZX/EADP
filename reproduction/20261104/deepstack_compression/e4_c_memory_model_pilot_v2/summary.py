"""Pure CPU stage-window summary; missing NVML values never become zero."""

def summarize(samples,metadata,boundaries,max_gap_ns):
 names=('release_ns','model_load_start_ns','model_load_end_ns','request_start_ns','request_end_ns','teardown_end_ns')
 if any(type(boundaries.get(k)) is not int for k in names) or [boundaries[k] for k in names]!=sorted(boundaries[k] for k in names) or any(boundaries[names[i]]>=boundaries[names[i+1]] for i in range(len(names)-1)):raise ValueError('invalid stage boundaries')
 if not samples or len(samples)!=len(metadata) or type(max_gap_ns) is not int or max_gap_ns<=0:raise ValueError('missing samples or invalid gap')
 last=None
 for s,m in zip(samples,metadata):
  a=s.get('query_start_ns');b=s.get('query_end_ns');c=m.get('query_start_ns');d=m.get('query_end_ns')
  if any(type(x) is not int for x in (a,b,c,d)) or not a<=c<=d<=b or (last is not None and (a<last or a-last>max_gap_ns)):raise ValueError('sample windows overlap, reorder or gap')
  if s.get('gpu_uuid')!=m.get('gpu_uuid') or m.get('status')!='completed':raise ValueError('metadata identity mismatch')
  last=b
 stages={'pre_cuda':(None,boundaries['release_ns']),'model_load':(boundaries['model_load_start_ns'],boundaries['model_load_end_ns']),'request':(boundaries['request_start_ns'],boundaries['request_end_ns']),'teardown':(boundaries['request_end_ns'],boundaries['teardown_end_ns'])}
 out={}
 for name,(start,end) in stages.items():
  selected=[(s,m) for s,m in zip(samples,metadata) if (start is None or s['query_start_ns']>=start) and s['query_end_ns']<=end]
  process=[];device=[];present=0;absent=0;present_unknown=0
  for s,m in selected:
   pv=s.get('used_bytes');dv=m.get('device_memory_bytes',{}).get('used')
   matching=[x for x in m.get('processes',[]) if x.get('pid')==s.get('host_pid') and x.get('type') in ('C','C+G')]
   if len(matching)>1:raise ValueError('duplicate target process')
   if matching:
    present+=1
    if matching[0].get('used_bytes') is None:present_unknown+=1
    if matching[0].get('used_bytes')!=pv:raise ValueError('target sample mismatch')
   else:
    absent+=1
    if pv is not None:raise ValueError('sample target absent from metadata')
   if any(v is not None and (type(v) is not int or v<0) for v in (pv,dv)):raise ValueError('invalid memory value')
   process.append(pv);device.append(dv)
  def coverage(values):
   known=[v for v in values if v is not None]
   return {'known':len(known),'unknown':len(values)-len(known),'max_known_bytes':max(known) if known else None}
  out[name]={'fully_contained_samples':len(selected),'process':dict(coverage(process),present=present,absent=absent,present_unknown=present_unknown),'device_used':coverage(device)}
 return {'status':'summarized','stages':out,'limitations':['unknown or absent worker is not zero','sampled maxima can miss transients','device and process totals are not interchangeable']}
