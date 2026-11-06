"""Pure 32-image paired timing statistics, with no I/O or authority."""
from collections import Counter,defaultdict
import numpy as np
TASKS={'DocVQA_VAL':11,'TextVQA_VAL':11,'AI2D_TEST':10};KS=(128,256,512,1024);PROTOCOLS=('natural','fixed32');SEED=20261105;B=10000

def summarize(jobs,events):
 if len(jobs)!=1520 or len(events)!=4560:raise ValueError('incomplete schedule/journal')
 groups=defaultdict(list);nphase=Counter();outcomes=defaultdict(Counter)
 for i,job in enumerate(jobs):
  segment=events[i*3:i*3+3]
  if [e.get('event') for e in segment]!=['job_started','raw_generation','job_completed'] or any(e.get('job')!=job for e in segment):raise ValueError('job ordering/identity')
  raw=segment[1]['raw'];rec=segment[2]['record']
  for k in ('task','qid','group_id','budget','protocol','output_ids','generation_scalars','min_new_tokens','eos_ids'):
   if raw.get(k)!=rec.get(k):raise ValueError('raw/record mismatch')
  bounds=rec['boundaries'];a,b,c=(bounds.get(k) for k in ('t0_before_image_read_ns','first_token_selected_ns','generation_complete_ns'))
  if any(type(v) is not int or v<0 for v in (a,b,c)) or not a<b<=c or raw.get('generation_complete_ns')!=c or bounds.get('first_criteria_length')!=1 or bounds.get('criteria_calls')!=len(rec['output_ids']) or bounds.get('clock_implementation')!='clock_gettime(CLOCK_MONOTONIC)' or bounds.get('explicit_sync_boundaries')!=['before_t0','first_token','generate_complete']:raise ValueError('time boundaries')
  nphase[job['phase']]+=1
  if job['phase']=='warmup':continue
  if job['phase']!='measure' or job['task'] not in TASKS or job['budget'] not in KS or job['protocol'] not in PROTOCOLS:raise ValueError('unknown job')
  key=(job['task'],job['protocol'],job['group_id'],str(job['qid']),job['budget']);groups[key].append((job['replicate'],(b-a)/1e6,(c-a)/1e6))
  outcomes[(job['task'],job['protocol'],job['budget'])].update({'eos':int(bool(rec['eos'])),'truncated':int(bool(rec['truncated'])),'tokens':len(rec['output_ids'])})
 if nphase!=Counter({'warmup':240,'measure':1280}):raise ValueError('warmup/measured denominator')
 if len(groups)!=32*8 or any(len(x)!=5 or len({r for r,_,_ in x})!=5 for x in groups.values()):raise ValueError('missing repeats')
 result={}
 for task,n in TASKS.items():
  byproto={};sets=[]
  for prot in PROTOCOLS:
   identities=sorted({(g,q) for t,p,g,q,k in groups if t==task and p==prot})
   if len(identities)!=n or len({g for g,q in identities})!=n:raise ValueError('task/group inventory')
   sets.append(identities);draw=np.random.default_rng(SEED).integers(0,n,size=(B,n));out={}
   values={}
   for k in KS:
    vals=[]
    for g,q in identities:
     rows=groups.get((task,prot,g,q,k))
     if rows is None or len(rows)!=5:raise ValueError('budget pairing missing')
     vals.append([float(np.median([x[1] for x in rows])),float(np.median([x[2] for x in rows]))])
    values[k]=np.array(vals)
   for k in KS:
    out[str(k)]={}
    for col,label in enumerate(('ttft_ms','generation_complete_ms')):
     v=values[k][:,col];ref=values[256][:,col]
     if np.any(v<=0) or np.any(ref<=0) or not np.isfinite(v).all():raise ValueError('invalid duration')
     diff=v-ref;ratio=v/ref
     def summary(x):return {'median':float(np.median(x)),'min':float(np.min(x)),'max':float(np.max(x)),'bootstrap_ci95':[float(y) for y in np.quantile(np.median(x[draw],axis=1),[.025,.975])]}
     out[str(k)][label]={'time':summary(v),'paired_difference_vs_k256_ms':summary(diff),'paired_ratio_vs_k256':summary(ratio)}
    out[str(k)]['counts']={'images':n,'runs':n*5,**dict(outcomes[(task,prot,k)])}
   byproto[prot]=out
  if sets[0]!=sets[1]:raise ValueError('protocol cohorts differ')
  result[task]=byproto
 return {'status':'completed','tasks':result,'jobs':1520,'warmups':240,'measured':1280,'images':32,'bootstrap':{'replicates':B,'seed':SEED,'unit':'image group','group_order':'ascending group_id then qid','numpy_version':np.__version__},'GT_opened':False,'scoring_performed':False,'limitations':['posthoc exploratory sample; three tasks reported separately','five timings reduced to one median per image, not independent observations','fixed32 is prefill plus 31 decode forwards','completion excludes token copy/fsync and text decoding','unknown background activity may affect samples; none removed']}
