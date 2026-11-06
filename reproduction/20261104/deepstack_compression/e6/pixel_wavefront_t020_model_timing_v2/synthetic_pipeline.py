"""No-GPU end-to-end fixture; retains every simulated scheduled result."""
import json,os
from pathlib import Path
from contract import schedule,validate_time,validate_nvml

def run_fake(anchors,reference,clock,out,mutate=None):
 out=Path(out);out.mkdir(exist_ok=False);results=[]
 for job in schedule(anchors):
  ref=reference[(job['task'],job['arm'])];a,b,c=clock(job)
  row=dict(job,**ref,start_monotonic_ns=a,first_token_monotonic_ns=b,completion_monotonic_ns=c,start_unix_ns=a+100000,completion_unix_ns=c+100000,first_token_synchronized=True,gpu_synchronized_before_start=True,all_samples_retained=True)
  if mutate:row=mutate(job,row)
  try:metrics=validate_time(job,row,ref);status='passed';error=None
  except ValueError as e:metrics=None;status='failed';error=str(e)
  results.append({'job':job,'record':row,'status':status,'metrics':metrics,'error':error})
  with (out/'journal.jsonl').open('a') as f:json.dump(results[-1],f);f.write('\n');f.flush();os.fsync(f.fileno())
 return results
