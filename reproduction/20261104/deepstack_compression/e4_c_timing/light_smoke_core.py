"""Instrumented 16-arm lightweight identity smoke core, no launch authority."""
import hashlib,json,os
from pathlib import Path
from light_smoke_plan import make_plan
from light_compare import compare_metadata,compare_archives

def key(record):
 if type(record.get('forced')) is not bool:raise ValueError('forced flag must be boolean')
 return (record['task'],str(record['qid']),record['forced'])
def append(path,record):
 with Path(path).open('a') as f:f.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
def indexed(records,fn):
 out={}
 for r in records:
  k=fn(r)
  if k in out:raise ValueError('duplicate diagnostic reference')
  out[k]=r
 return out

def run_diagnostics(torch,w,pruner,rows,anchors,b_records,old_on,out,generate,fresh_runner,light_runner,archive_factory,forced_token):
 plan=make_plan(anchors);out=Path(out)
 if not out.is_dir() or any((out/x).exists() for x in ('generated.jsonl','records.jsonl','comparisons.jsonl','errors.jsonl')):raise ValueError('invalid or reused diagnostic output')
 inputs=indexed(rows,lambda x:(x['task'],str(x['qid'])));b=indexed(b_records,lambda x:(*key(x),x['mode']));old=indexed(old_on,key)
 if len(inputs)!=3 or len(b)!=8 or len(old)!=4:raise ValueError('reference inventory')
 expected={(j['task'],j['qid']) for j in plan['jobs']}
 if set(inputs)!=expected:raise ValueError('anchor/input inventory')
 fresh={}
 for job in plan['jobs']:
  row=inputs[(job['task'],job['qid'])];case=(job['task'],job['qid'],job['forced']);mode=job['path']
  if row.get('split')!='DEV' or row.get('group_id')!=job['group_id']:raise ValueError('anchor binding')
  token=forced_token if job['forced'] else None
  name=hashlib.sha256(json.dumps([*case,mode],separators=(',',':')).encode()).hexdigest()
  archive=archive_factory(name)
  def journal(record):append(out/'generated.jsonl',dict(record,path=mode))
  try:
   if mode=='fresh_on_reference':
    record=fresh_runner(torch,w,pruner,row,256,True,generate,journal,archive,token,capture_prefill_logits=True)
   else:record=light_runner(torch,w,pruner,row,mode,generate,journal,archive,token)
   append(out/'records.jsonl',{'job':job,'record':record})
   if mode=='fresh_on_reference':
    fresh[case]=record;legacy=old[case]
    normal=dict(record,mode='C-ON',ds_none=[False]*3)
    allow=legacy.get('prefill_logits') is None
    metadata=compare_metadata(normal,legacy,allow_missing_reference_logits=allow,allow_missing_reference_kv=True)
    arrays=compare_archives(normal,legacy,allow_missing_reference_logits=allow)
    reference='accepted_e4_on'
   else:
    reference_record=fresh[case] if mode=='C-ON' else b[(*case,mode)]
    metadata=compare_metadata(record,reference_record);arrays=compare_archives(record,reference_record)
    reference='fresh_on' if mode=='C-ON' else 'accepted_b_'+mode
   append(out/'comparisons.jsonl',{'job':job,'reference':reference,'metadata':metadata,'arrays':arrays})
  except BaseException as exc:
   append(out/'errors.jsonl',{'job':job,'error_type':type(exc).__name__,'error':str(exc)})
   raise
 return {'status':'completed','jobs':len(plan['jobs']),'scope':plan['scope']}
