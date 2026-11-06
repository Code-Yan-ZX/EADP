"""Model-backed random smoke continuation; deliberately has no CLI/authority.

Only a future independently reviewed launcher may call this module after
complete CPU/preflight admission, source pinning, GPU isolation and full E1
admission. Existing E1 mixed DEV journal is parsed to extract identity fields;
no new labels are opened and no scores are computed.
"""
import hashlib,json,os,time,traceback
from pathlib import Path

def run(prepared,torch,native,wrapper_class,load_refs,receipt,plan,manifest,out,generate_method,selector,budget_runner,random_adapter,cache_factory,identity,run_arm,inventory,diagnostic_run,preflight_one,source_check,versions):
 out=Path(out)
 if not out.is_dir():raise ValueError('launcher output missing')
 rows=[]
 for a in plan['anchors']:
  found=[r for r in prepared['rows'] if r['task']==a['task'] and str(r['qid'])==str(a['qid'])]
  if len(found)!=1 or any(str(found[0].get(k))!=str(a.get(k)) for k in ('task','qid','group_id','image_sha256','message_sha256')):raise ValueError('anchor not bound')
  rows.append(found[0])
 references=load_refs(receipt,rows)
 inventory.validate_inventory(rows,references,manifest,plan['anchors'])
 if torch.cuda.get_device_name(0)!='NVIDIA A40':raise ValueError('unexpected GPU')
 torch.cuda.set_per_process_memory_fraction(.9,0)
 start=time.monotonic()
 def progress(event,**kw):
  with (out/'progress.jsonl').open('a') as f:json.dump({'event':event,'unix':time.time(),**kw},f);f.write('\n');f.flush();os.fsync(f.fileno())
 def save(name,value):
  with (out/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 progress('model_loading')
 try:
  wrapper=wrapper_class(**prepared['plan']['recipe'])
  if wrapper.model.dtype!=torch.bfloat16 or wrapper.model.config._attn_implementation!='flash_attention_2':raise ValueError('actual model backend mismatch')
  for row in rows:
   key=(row['task'],str(row['qid']));preflight_one(torch,wrapper,row,prepared['refs'][key])
  save('model_ready.json',{'model_load_seconds':time.monotonic()-start,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'prediction_scope':'accepted E1 DEV mixed journal parsed for identity only','extra_author_greedy':True,'timing_eligible':False,'recipe':prepared['plan']['recipe'],'versions':versions,'weight_stats_before':prepared['weights_before'],'e1_acceptance_sha256':prepared.get('e1_acceptance_sha256')})
  def generate(w,row):
   with torch.inference_mode():return generate_method(w,row['message'],dataset=row['task'])
  progress('model_ready')
  counter={'steps':0}
  def heartbeat(_module,_args,_kwargs,_output):
   counter['steps']+=1
   if counter['steps']%32==0:progress('forward_progress',steps=counter['steps'])
  hook=wrapper.model.model.language_model.register_forward_hook(heartbeat,with_kwargs=True)
  def before_complete():
   source=source_check()
   after={}
   for path,before in prepared['weights_before'].items():
    stat=Path(path).stat();after[path]=[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]
    if after[path]!=before:raise ValueError('weight identity changed: '+path)
   save('completion_provenance.json',{'status':'passed','weight_stats_before':prepared['weights_before'],'weight_stats_after':after,'source_sha256':source['source_sha256'],'GT_opened':False,'scoring_performed':False})
  try:
   result=diagnostic_run.run(torch,wrapper,native,rows,references,manifest,plan['anchors'],out,generate,selector,budget_runner,random_adapter,cache_factory,identity,run_arm,inventory,before_complete)
  finally:hook.remove()
  progress('diagnostic_completed')
  return result
 except BaseException as exc:
  progress('failed',error_type=type(exc).__name__)
  if not (out/'failure.json').exists():
   save('failure.json',{'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'GT_opened':False,'scoring_performed':False})
  raise
