"""Independent, CPU-only oracle for normalized endpoint and schedule records.

This does not call a model or grant GPU admission. A candidate integration
must map its real records to this schema without discarding anything.
"""
from collections import Counter,defaultdict
ARMS=('rres','eadp');REPS=5
def key(x):return (x['task'],str(x['qid']),x['group_id'])
def normalize(arm,full,prompt):
 if arm not in ARMS or not isinstance(full,list) or any(type(x) is not int for x in full):raise ValueError('arm or tokens')
 if arm=='rres':
  if not isinstance(prompt,list) or not prompt or full[:len(prompt)]!=prompt:raise ValueError('R-res prompt prefix')
  tokens=full[len(prompt):]
 else:tokens=full
 if not tokens:raise ValueError('empty continuation')
 return tokens
def endpoint(record):
 for n in ('t0','first','generation_done','response_done'):
  if type(record.get(n)) is not int or record[n]<0:raise ValueError('invalid endpoint '+n)
 if not record['t0']<record['first']<=record['generation_done']<=record['response_done']:raise ValueError('endpoint order')
 if record.get('first_calls')!=1 or record.get('generation_calls')!=1 or record.get('persist_before_response') is not False:raise ValueError('missing endpoint or early persistence')
 if not record.get('time_namespace') or not record.get('clock'):raise ValueError('unmatched clock identity')
 tokens=normalize(record['arm'],record['raw'],record.get('prompt'))
 if tokens!=record['same_arm_expected']:raise ValueError('same-arm token identity')
 if not record.get('eos_ids') or record['eos']!=(tokens[-1] in record['eos_ids']) or record['truncated']!=(tokens[-1] not in record['eos_ids']):raise ValueError('EOS state')
 return tokens
def schedule(jobs,measured,warmups):
 """Require normalized jobs with sequence, process, task, group, arm, phase, rep, block."""
 ms={key(x) for x in measured};ws={key(x) for x in warmups}
 if len(ms)!=32 or len(ws)!=30 or len(ms)!=len(measured) or len(ws)!=len(warmups):raise ValueError('input inventory')
 if {x[2] for x in ms}&{x[2] for x in ws} or len({x[2] for x in ms})!=32 or len({x[2] for x in ws})!=30:raise ValueError('image group overlap')
 if Counter(x[0] for x in ms)!={'DocVQA_VAL':11,'TextVQA_VAL':11,'AI2D_TEST':10} or any(v!=10 for v in Counter(x[0] for x in ws).values()) or {x[0] for x in ms}!={x[0] for x in ws}:raise ValueError('task inventory')
 if len(jobs)!=380 or [x['order'] for x in jobs]!=list(range(380)):raise ValueError('job count/order')
 actual=Counter();processes=defaultdict(list);assign={}
 for j in jobs:
  k=key(j);arm=j['arm'];phase=j['phase'];rep=j['rep'];block=j['block'];proc=j['process']
  if arm not in ARMS or phase not in ('warmup','measure') or type(rep) is not int or block not in (0,1):raise ValueError('invalid job')
  if (phase=='warmup' and (k not in ws or rep!=0)) or (phase=='measure' and (k not in ms or rep not in range(REPS))):raise ValueError('wrong phase or repetition')
  target=(k,arm,phase,rep);actual[target]+=1;processes[proc].append(j)
  old=assign.setdefault((k,arm),block)
  if old!=block:raise ValueError('image split across blocks')
 expected=Counter((k,a,'measure',r) for k in ms for a in ARMS for r in range(REPS));expected.update((k,a,'warmup',0) for k in ws for a in ARMS)
 if actual!=expected:raise ValueError('missing/duplicate job')
 if any(assign[(k,'rres')]!=assign[(k,'eadp')] for k in ms|ws):raise ValueError('unpaired block')
 if len(processes)!=12:raise ValueError('process inventory')
 signatures=[]
 for rows in processes.values():
  sig={(r['task'],r['arm'],r['block']) for r in rows}
  if len(sig)!=1:raise ValueError('mixed process')
  signatures.append(next(iter(sig)))
  if sum(r['phase']=='warmup' for r in rows)!=5:raise ValueError('warmup inventory')
  positions=[r['order'] for r in rows]
  if positions!=list(range(min(positions),max(positions)+1)) or any(r['phase']=='warmup' for r in rows[5:]) or any(r['phase']!='warmup' for r in rows[:5]):raise ValueError('noncontiguous process or late warmup')
 if len(set(signatures))!=12:raise ValueError('duplicate process')
 return {'jobs':len(jobs),'measured':320,'warmups':60,'processes':12}
def flatten_processes(processes):
 """Adapt proposed list-of-processes schema without omitting requests."""
 if len(processes)!=12 or [p['order'] for p in processes]!=list(range(12)):raise ValueError('process order')
 out=[]
 for i,p in enumerate(processes):
  if p['arm'] not in ARMS or p['block'] not in (0,1) or not isinstance(p['requests'],list):raise ValueError('process fields')
  for r in p['requests']:
   if r['task']!=p['task'] or r['phase'] not in ('warmup','measured'):raise ValueError('request process identity')
   out.append({'order':len(out),'process':i,'task':r['task'],'qid':r['qid'],'group_id':r['group_id'],'arm':p['arm'],'block':p['block'],'phase':'measure' if r['phase']=='measured' else 'warmup','rep':r['replicate']})
 return out
