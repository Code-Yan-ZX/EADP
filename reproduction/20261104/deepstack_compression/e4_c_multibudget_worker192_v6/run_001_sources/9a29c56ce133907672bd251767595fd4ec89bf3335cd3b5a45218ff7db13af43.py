"""Complete 192-image timing core: pure orchestration, never a GPU entrypoint."""
from collections import Counter
KS=(128,256,512,1024);PROTOCOLS=('natural','fixed32')
def prepare(proposal,rows,panel,fixed):
 if proposal.get('GPU_authorized') is not False or proposal.get('GT_opened') is not False or proposal.get('status')!='proposal_frozen':raise ValueError('proposal not admitted')
 expected=proposal['measurement_identities']+proposal['warmup_identities'];jobs=proposal['jobs']
 if len(expected)!=222 or len(jobs)!=7920 or len(rows)!=222:raise ValueError('frozen inventory')
 keyed=lambda r:(r['task'],str(r['qid']));by={keyed(r):r for r in rows};identity={keyed(r):r for r in expected}
 if len(by)!=222 or len(identity)!=222 or set(by)!=set(identity) or len({(x['task'],x['group_id']) for x in rows})!=222 or not isinstance(fixed,dict) or len(fixed)!=888 or len(panel)!=888:raise ValueError('input/reference inventory')
 measured={(x['task'],str(x['qid'])) for x in proposal['measurement_identities']};warm={(x['task'],str(x['qid'])) for x in proposal['warmup_identities']}
 if len(measured)!=192 or len(warm)!=30 or measured&warm or Counter(t for t,q in measured)!={'DocVQA_VAL':64,'TextVQA_VAL':64,'AI2D_TEST':64} or Counter(t for t,q in warm)!={'DocVQA_VAL':10,'TextVQA_VAL':10,'AI2D_TEST':10}:raise ValueError('cohort overlap or task denominator')
 for key,row in by.items():
  sample=identity[key]
  if row.get('split')!='DEV' or any(str(row.get(t))!=str(sample.get(t)) for t in ('task','qid','group_id','image_path','image_sha256','message_sha256')) or row.get('processor_expected')!=sample.get('processor_reference') or row.get('image_start')!=sample.get('image_start_reference'):raise ValueError('input identity')
  for k in KS:
   p=panel[(*key,k)];f=fixed[(*key,k)]
   if p.get('budget')!=k or p.get('ds_on') is not True or any(str(p.get(t))!=str(row.get(t)) for t in ('task','qid','group_id','image_sha256','message_sha256')) or p.get('processor')!=row.get('processor_expected') or p.get('image_start')!=row.get('image_start'):raise ValueError('natural reference identity')
   if any(str(f.get(t))!=str(row.get(t)) for t in ('task','qid','group_id','image_sha256','message_sha256')) or f.get('budget')!=k or f.get('protocol')!='fixed32' or f.get('min_new_tokens')!=32 or f.get('generation_scalars')!=dict(p['generation_scalars'],max_new_tokens=32) or f.get('eos_ids')!=p['eos_ids'] or not isinstance(f.get('answer'),str):raise ValueError('fixed32 reference identity')
   ids=f.get('output_ids');eos=f['eos_ids']
   if not isinstance(ids,list) or len(ids)!=32 or any(type(x) is not int for x in ids) or not eos or any(x in eos for x in ids[:-1]):raise ValueError('fixed32 output')
 seen=Counter()
 for i,j in enumerate(jobs):
  key=(j.get('task'),str(j.get('qid')));phase=j.get('phase');reps=range(5) if phase=='measure' else (0,) if phase=='warmup' else ()
  if j.get('order')!=i or key not in by or j.get('group_id')!=by[key]['group_id'] or j.get('budget') not in KS or j.get('protocol') not in PROTOCOLS or type(j.get('replicate')) is not int or j['replicate'] not in reps or (phase=='measure' and key not in measured) or (phase=='warmup' and key not in warm):raise ValueError('job identity/schedule')
  seen[(key,phase,j['replicate'],j['budget'],j['protocol'])]+=1
 if len(seen)!=7920 or any(n!=1 for n in seen.values()) or sum(j['phase']=='measure' for j in jobs)!=7680 or sum(j['phase']=='warmup' for j in jobs)!=240:raise ValueError('job schedule denominator')
 return by

def run(proposal,rows,panel,fixed,execute,emit):
 by=prepare(proposal,rows,panel,fixed);done=0;measured_fixed={}
 for job in proposal['jobs']:
  key=(job['task'],str(job['qid']));p=panel[(*key,job['budget'])];emit({'event':'job_started','job':job});seen=[]
  try:
   def raw(value):
    if seen:raise ValueError('multiple generated exposures')
    emit({'event':'raw_generation','job':job,'raw':value});seen.append(value)
   rec=execute(by[key],job['budget'],job['protocol'],p,raw)
   # Preserve actual result before validation, including failures.
   emit({'event':'job_completed','job':job,'record':rec})
   if len(seen)!=1 or any(str(rec.get(k))!=str(job.get(k)) for k in ('task','qid','group_id','budget','protocol')) or any(rec.get(k)!=seen[0].get(k) for k in ('task','qid','group_id','budget','protocol','output_ids','eos_ids','generation_scalars','min_new_tokens')):raise ValueError('generated/result identity')
   if job['protocol']=='fixed32':
    f=fixed[(*key,job['budget'])]
    if any(rec.get(k)!=f.get(k) for k in ('output_ids','answer','eos_ids','generation_scalars','min_new_tokens')):raise ValueError('fixed32 reference mismatch')
    if job['phase']=='measure':measured_fixed[(*key,job['budget'])]=True
   else:
    if any(rec.get(k)!=p.get(k) for k in ('output_ids','answer','eos_ids','generation_scalars','eos','truncated')):raise ValueError('natural E4 endpoint mismatch')
   done+=1
  except BaseException as error:
   emit({'event':'job_failed','job':job,'error_type':type(error).__name__,'error':str(error)});raise
 if done!=7920 or len(measured_fixed)!=768:raise ValueError('completion denominator')
 return {'status':'passed','jobs':done,'warmups':240,'measured':7680,'fixed32_measure_keys':768,'GT_opened':False,'scoring_performed':False}
