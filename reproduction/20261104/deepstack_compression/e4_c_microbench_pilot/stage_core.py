"""Prepared component-pilot orchestration; no standalone GPU admission."""
from capture_guard import capture_one
from identity_contract import validate
from contracts import run_jobs

def execute(torch,wrapper,native,row,ref,schedule,expected,preflight,author_generate,Adapter,Capture,clone,tensor_meta,equal,fingerprint,event_factory,synchronize,stream_is_expected,journal,archive):
 if row.get('split')!='DEV' or any(str(row.get(k))!=str(ref.get(k)) for k in ('task','qid','group_id','message_sha256','image_start')) or len(schedule.get('images',[]))!=1 or tuple(schedule['images'][0])!=(row['task'],row['group_id']):raise ValueError('pilot input identity')
 # Processor identity must pass before even a one-token diagnostic exposure.
 preflight(torch,wrapper,row,ref)
 with torch.inference_mode():
  operations,references,actual,hooks_clean=capture_one(torch,wrapper,native,row,author_generate,Adapter,Capture,clone,journal)
  evidence=validate(actual,ref,tensor_meta)
  if not hooks_clean():raise ValueError('hooks returned after capture')
  # Archival is wholly outside the measurement interval. A caller must record
  # paths/hashes and later verify them; this function does not grant GPU access.
  archive(actual,references);journal({'status':'capture_identity_passed','identity':evidence,'task':row['task'],'qid':str(row['qid']),'group_id':row['group_id']})
  return run_jobs(schedule,expected,operations,references,event_factory,synchronize,equal,fingerprint,journal,hooks_clean,torch.is_inference_mode_enabled,stream_is_expected)
if __name__=='__main__':raise SystemExit('GPU worker, bounded launcher and admission are not implemented')
