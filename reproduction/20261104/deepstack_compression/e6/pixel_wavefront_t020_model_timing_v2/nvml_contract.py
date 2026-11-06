"""CPU validation of existing sanitized NVML metadata; never queries a GPU."""
MIB=1048576
def validate(metadata,target_pid,uuid,window,max_gap_ns):
 if not metadata or type(target_pid) is not int or target_pid<=0 or not uuid or type(max_gap_ns) is not int or max_gap_ns<=0:raise ValueError('sampling identity')
 begin,end=window
 if type(begin) is not int or type(end) is not int or not 0<begin<end:raise ValueError('request interval')
 previous=None;out=[]
 for s in metadata:
  a,b=s.get('query_start_ns'),s.get('query_end_ns')
  if s.get('status')!='completed' or s.get('gpu_uuid')!=uuid or type(a) is not int or type(b) is not int or a<=0 or b<a or b-a>max_gap_ns or (previous is not None and (a<previous[1] or b-previous[0]>max_gap_ns)):raise ValueError('query status, gap or clock')
  memory=s.get('device_memory_bytes');procs=s.get('processes')
  if not isinstance(memory,dict) or set(memory)!={'total','used','free','reserved'} or not isinstance(procs,list):raise ValueError('sample inventory')
  def metric(x):
   if x is not None and (type(x) is not int or x<0 or x%MIB):raise ValueError('invalid MiB-granularity value')
   return x
  for v in memory.values():metric(v)
  seen=set();target=[];background=[]
  for p in procs:
   pid,kind=p.get('pid'),p.get('type')
   if type(pid) is not int or pid<=0 or pid in seen or kind not in ('C','G','C+G'):raise ValueError('invalid process identity')
   seen.add(pid);metric(p.get('used_bytes'))
   if pid==target_pid and kind in ('C','C+G'):target.append(p)
   else:background.append(p)
  if len(target)>1 or any(p['type'] in ('C','C+G') for p in background):raise ValueError('other compute process or duplicate target')
  out.append({'request_overlap':a<=end and b>=begin,'target_present':bool(target),'target_used_bytes':target[0]['used_bytes'] if target else None,'device_memory_bytes':memory,'background_graphics_pids':[p['pid'] for p in background if p['type']=='G']});previous=(a,b)
 if metadata[0]['query_start_ns']>begin or metadata[-1]['query_end_ns']<end:raise ValueError('request not bracketed')
 return out
