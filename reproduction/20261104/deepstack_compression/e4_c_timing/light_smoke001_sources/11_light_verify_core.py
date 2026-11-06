"""Read completed DEV diagnostic journals and replay all identity comparisons.

No launcher/terminal admission or proof writing occurs here. An independent
outer verifier must first check true completed exit, source and input identities.
"""
import json
from pathlib import Path
from light_compare import compare_metadata,compare_archives
from light_smoke_core import indexed,key
from light_smoke_plan import make_plan
from light_verify_records import validate_light

def lines(path):
 with Path(path).open() as f:return [json.loads(x) for x in f if x.strip()]

def verify_records(out,ref,forced_token):
 out=Path(out).resolve();plan=make_plan(ref['plan']['anchors']);jobs=plan['jobs']
 if ref.get('status')!='passed' or ref.get('plan')!=plan:raise ValueError('reference plan mismatch')
 records=lines(out/'records.jsonl');generated=lines(out/'generated.jsonl');comparisons=lines(out/'comparisons.jsonl')
 if len(records)!=16 or len(generated)!=16 or len(comparisons)!=16 or (out/'errors.jsonl').exists():raise ValueError('incomplete diagnostic journals')
 b=indexed(ref['b_records'],lambda x:(*key(x),x['mode']));old=indexed(ref['old_on'],key)
 if len(b)!=8 or len(old)!=4:raise ValueError('reference inventory mismatch')
 fresh={};archive_paths=set();light_count=0
 for job,item,raw,stored in zip(jobs,records,generated,comparisons):
  if item.get('job')!=job or stored.get('job')!=job:raise ValueError('journal order mismatch')
  r=item['record'];case=(job['task'],job['qid'],job['forced']);mode=job['path']
  if type(r.get('forced')) is not bool or type(raw.get('forced')) is not bool:raise ValueError('generation forced flag must be boolean')
  for field in ('task','qid','group_id','forced','output_ids','eos_ids','generation_scalars'):
   expected=job[field] if field in ('task','qid','group_id','forced') else r[field]
   if raw.get(field)!=expected or (field in ('task','qid','group_id','forced') and r.get(field)!=expected):raise ValueError('generation journal mismatch: '+field)
  if raw.get('path')!=mode:raise ValueError('generation path mismatch')
  path=Path(r['arrays']['path']).resolve()
  if path.parent!=(out/'arrays').resolve() or path in archive_paths:raise ValueError('invalid or repeated diagnostic archive')
  archive_paths.add(path)
  if mode=='fresh_on_reference':
   fresh[case]=r;legacy=old[case];actual=dict(r,mode='C-ON',ds_none=[False]*3)
   allow=legacy.get('prefill_logits') is None;reference_name='accepted_e4_on'
  else:
   if r.get('mode')!=mode:raise ValueError('record mode mismatch')
   if raw.get('min_new_tokens')!=r.get('min_new_tokens'):raise ValueError('generation minimum mismatch')
   validate_light(r,path,forced_token);light_count+=1
   actual=r;legacy=fresh[case] if mode=='C-ON' else b[(*case,mode)];allow=False
   reference_name='fresh_on' if mode=='C-ON' else 'accepted_b_'+mode
  meta=compare_metadata(actual,legacy,allow_missing_reference_logits=allow,allow_missing_reference_kv=mode=='fresh_on_reference')
  arrays=compare_archives(actual,legacy,allow_missing_reference_logits=allow)
  if stored!={'job':job,'reference':reference_name,'metadata':meta,'arrays':arrays}:raise ValueError('stored comparison differs from replay')
 if light_count!=12 or len(archive_paths)!=16:raise ValueError('lightweight generation count mismatch')
 return {'status':'passed','generations':len(records),'lightweight_generations':light_count,'comparisons':len(comparisons),'GT_opened':False,'scoring_performed':False,'timing_claims':False}
