"""CPU classification of monitoring query intervals; never drop timing rows."""
def classify(start,end,namespace,queries):
 if type(start) is not int or type(end) is not int or not 0<start<end or not isinstance(namespace,str) or not namespace:raise ValueError('measurement window')
 if queries is None:return {'status':'unknown','overlap_indices':[],'reason':'query coverage absent'}
 if not isinstance(queries,list):raise ValueError('malformed queries')
 if not queries:return {'status':'unknown','overlap_indices':[],'reason':'no recorded queries'}
 for q in queries:
  if (not isinstance(q,dict) or type(q.get('start_ns')) is not int or type(q.get('end_ns')) is not int
      or not 0<q['start_ns']<=q['end_ns'] or not isinstance(q.get('time_namespace'),str) or not q['time_namespace']):raise ValueError('malformed query window')
 if any(q['time_namespace']!=namespace for q in queries):return {'status':'unknown','overlap_indices':[],'reason':'time namespace not proven identical'}
 return {'status':'observed','overlap_indices':[i for i,q in enumerate(queries) if q['start_ns']<=end and q['end_ns']>=start], 'reason':None,'background_activity_outside_queries':'unknown'}
