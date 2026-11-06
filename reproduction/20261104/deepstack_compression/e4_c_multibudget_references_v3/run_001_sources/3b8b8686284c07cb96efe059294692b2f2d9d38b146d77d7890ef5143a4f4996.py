"""Single-model 1520-job continuation; not a GPU entrypoint or admission gate.

Caller must independently validate and pin all files, accepted panel/E1 proofs,
processor images, fixed32 identity, supervisor, Docker, deadline and GPU before
loading the model or calling this module.  Every job output is journaled.
"""
import json,os
from pathlib import Path
BUDGETS=(128,256,512,1024)

def prepare(plan,rows,records,contract):
 identities=plan['input_identities'];jobs=plan['jobs']
 if plan.get('GPU_authorized') is not False or plan.get('GT_opened') is not False or len(identities)!=62 or len(jobs)!=1520:raise ValueError('frozen inventory incomplete')
 by={(r['task'],str(r['qid'])):r for r in rows};expected={}
 if len(by)!=62:raise ValueError('accepted input rows not unique')
 for item in identities:
  key=(item['task'],str(item['qid']))
  if key in expected or key not in by:raise ValueError('sample identity duplicate or absent')
  row=by[key]
  if row.get('split')!='DEV' or any(item.get(k)!=row.get(k) for k in ('task','group_id','image_path','image_sha256','message_sha256')):raise ValueError('frozen sample/E1 identity mismatch')
  expected[key]=item
  for k in BUDGETS:
   p=records[(key[0],key[1],k)]
   if p.get('budget')!=k or p.get('ds_on') is not True or any(str(p.get(t))!=str(row.get(t)) for t in ('task','qid','group_id','image_sha256','message_sha256')) or p.get('processor')!=row.get('processor_expected') or p.get('image_start')!=row.get('image_start'):raise ValueError('sample/E4 endpoint mismatch')
 cohort_source=plan['_validated_full_cohort'] if '_validated_full_cohort' in plan else None
 if cohort_source is None:raise ValueError('full 64/10 source required for independent schedule rebuild')
 bygroup={(r['task'],r['group_id']):r for r in rows}
 if len(bygroup)!=62:raise ValueError('input image groups not unique')
 # The original 64 measured source inventory is needed to verify the frozen
 # selected groups, but identities are required only for selected and warmup.
 inventory,reconstructed=contract.build(cohort_source,bygroup)
 if inventory!=plan['cohort'] or reconstructed!=jobs:raise ValueError('frozen randomized schedule mismatch')
 if sum(j['phase']=='warmup' for j in jobs)!=240 or sum(j['phase']=='measure' for j in jobs)!=1280:raise ValueError('phase denominator')
 return by

def run(plan,rows,records,fixed_references,contract,execute,emit):
 """Execute all jobs or stop at first error; no silent dropping or resume."""
 by=prepare(plan,rows,records,contract);completed=0;fixed_refs={}
 if not isinstance(fixed_references,dict) or len(fixed_references)!=248:raise ValueError('independent fixed32 references incomplete')
 for key,row in by.items():
  for budget in BUDGETS:
   ref=fixed_references.get((key[0],key[1],budget));panel=records[(key[0],key[1],budget)]
   if not isinstance(ref,dict) or any(str(ref.get(x))!=str(row.get(x)) for x in ('task','qid','group_id','image_sha256','message_sha256')) or ref.get('budget')!=budget or ref.get('protocol')!='fixed32' or ref.get('min_new_tokens')!=32 or ref.get('generation_scalars')!=dict(panel['generation_scalars'],max_new_tokens=32) or ref.get('eos_ids')!=panel['eos_ids'] or not isinstance(ref.get('answer'),str):raise ValueError('independent fixed32 reference identity or recipe mismatch')
   ids=ref.get('output_ids');eos=ref['eos_ids']
   if not isinstance(ids,list) or len(ids)!=32 or any(type(x) is not int for x in ids) or not eos or any(x in eos for x in ids[:-1]):raise ValueError('invalid independent fixed32 reference output')
 for job in plan['jobs']:
  row=by[(job['task'],str(job['qid']))];panel=records[(job['task'],str(job['qid']),job['budget'])]
  emit({'event':'job_started','job':job})
  try:
   def raw(value):emit({'event':'raw_generation','job':job,'raw':value})
   rec=execute(row,job['budget'],job['protocol'],panel,raw)
   if rec.get('task')!=job['task'] or str(rec.get('qid'))!=str(job['qid']) or rec.get('group_id')!=job['group_id'] or rec.get('budget')!=job['budget'] or rec.get('protocol')!=job['protocol']:raise ValueError('actual result identity mismatch')
   # Record before repeat check, so a mismatch is preserved and never omitted.
   emit({'event':'job_completed','job':job,'record':rec});completed+=1
   if job['phase']=='measure' and job['protocol']=='fixed32':
    key=(job['task'],str(job['qid']),job['budget']);signature=(rec['output_ids'],rec['answer'],rec['eos_ids'])
    if key in fixed_refs and fixed_refs[key]!=signature:raise ValueError('fixed32 same-sample repeat mismatch')
    fixed_refs[key]=signature
   if job['protocol']=='fixed32':
    ref=fixed_references[(job['task'],str(job['qid']),job['budget'])]
    if any(rec.get(k)!=ref.get(k) for k in ('output_ids','answer','eos_ids','generation_scalars','min_new_tokens')):raise ValueError('fixed32 independent reference mismatch')
  except BaseException as e:
   emit({'event':'job_failed','job':job,'error_type':type(e).__name__,'error':str(e)});raise
 if completed!=1520 or len(fixed_refs)!=128:raise ValueError('actual completion denominator mismatch')
 return {'status':'passed','jobs':completed,'warmup_jobs':240,'measured_jobs':1280,'fixed32_measure_keys':128,'GT_opened':False,'scoring_performed':False}

def append(path,record):
 with Path(path).open('a') as f:f.write(json.dumps(record,sort_keys=True,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())

if __name__=='__main__':raise SystemExit('GPU locked: worker admission/launcher/real-model fixed32 identity are not yet available')
