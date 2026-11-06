"""Complete inventory validation; failed cases remain in the denominator."""
import math
from trials import measurement_order
BUDGETS=(128,256,512)
def partition(events,pilot):
 if len(pilot)!=32 or not events or events[0].get('status')!='synthetic_completed':raise ValueError('missing pilot or synthetic evidence')
 chunks=[];cur=None
 for e in events[1:]:
  i=e.get('image_index');status=e.get('status')
  if status=='image_started':
   if cur is not None or i!=len(chunks):raise ValueError('image order')
   cur=[e]
  elif status=='image_completed':
   if cur is None or i!=len(chunks):raise ValueError('image completion')
   cur.append(e);chunks.append(cur);cur=None
  else:
   if cur is None or i!=len(chunks):raise ValueError('orphan event')
   cur.append(e)
 if cur is not None or len(chunks)!=32:raise ValueError('incomplete 32-image run')
 for row,chunk in zip(pilot,chunks):
  for e in (chunk[0],chunk[-1]):
   if any(str(e.get(k))!=str(row[k]) for k in ('task','qid','group_id')):raise ValueError('image identity')
  complete=[e for e in chunk if e.get('status')=='case_completed']
  if [x.get('budget') for x in complete]!=list(BUDGETS):raise ValueError('missing/duplicated budget cases')
  for case in complete:
   k=case['budget'];r=case.get('result');trace=[e['measurement'] for e in chunk if e.get('status')=='case_measurement' and e.get('budget')==k]
   if r=='not_evaluated':
    if not any(e.get('status')=='capture_failed' for e in chunk) or trace:raise ValueError('invalid capture failure')
   elif not isinstance(r,dict) or r.get('status') not in ('measured','not_evaluated'):raise ValueError('unknown case result')
   elif r['status']=='measured':
    expected=[(n,name) for n in range(5) for name in measurement_order(n)]
    if r.get('measurements')!=trace or [(m.get('repeat'),m.get('implementation')) for m in trace]!=expected:raise ValueError('measurement order/inventory')
   elif len(trace)>15:raise ValueError('invalid failed measurement trace')
 return chunks

def validate_candidate_ids(case,original,synthetic_allowed):
 if case.get('status')!='measured':return
 original=list(original);identity=case.get('identity',{});measured=case.get('measurements',[])
 if identity.get('author_equals_endpoint') is not True:raise ValueError('author preliminary mismatch')
 allowed={'author':True}
 for name in ('fused','graph'):
  initial=identity.get(name+'_ordered_ids');flag=identity.get(name+'_equals_author')
  if (initial is None and flag is not None) or (initial is not None and flag is not (initial==original)) or (not synthetic_allowed[name] and initial is not None):raise ValueError('initial candidate identity mismatch')
  allowed[name]=bool(flag) and synthetic_allowed[name]
 outcomes={'fused':allowed['fused'],'graph':allowed['graph']}
 for m in measured:
  name=m.get('implementation');state=m.get('status')
  if name not in allowed:raise ValueError('unknown method')
  if state=='unavailable':
   if allowed[name] or not isinstance(m.get('reason'),str):raise ValueError('unjustified unavailable method')
   if name in outcomes:outcomes[name]=False
   continue
  if state!='measured' or not allowed[name]:raise ValueError('timing before identity gate')
  if any(type(m.get(k)) not in (int,float) or not math.isfinite(m[k]) or m[k]<0 for k in ('gpu_event_ms','host_wall_ms')):raise ValueError('invalid timing')
  equal=m.get('ordered_ids')==original
  if m.get('equals_author') is not equal:raise ValueError('selection flag mismatch')
  if not equal:
   allowed[name]=False
   if name in outcomes:outcomes[name]=False
   if name=='author':allowed['fused']=allowed['graph']=False;outcomes={'fused':False,'graph':False}
 for name in ('fused','graph'):
  related=[m for m in measured if m.get('implementation')==name]
  expected=outcomes[name] and len(related)==5 and all(m.get('equals_author') is True for m in related)
  if case.get('candidate_matches_author',{}).get(name)!=expected:raise ValueError('candidate result mismatch')
