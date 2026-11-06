"""Prospective sequential nine-job scheduler; GPU gate currently hard locked."""
import argparse,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];PY='/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python'
def append(path,value):
 with Path(path).open('a') as f:f.write(json.dumps(value,separators=(',',':'))+'\n');f.flush();os.fsync(f.fileno())
def run_all(jobs,run_one,record):
 if len(jobs)!=9 or [j['order'] for j in jobs]!=list(range(9)):raise ValueError('frozen nine-job order')
 for job in jobs:
  record({'event':'started','job':job})
  try:
   value=run_one(job)
   if value.get('status')!='passed' or value.get('job')!=job:raise ValueError('job result invalid')
   record({'event':'completed','job':job,'result':value})
  except BaseException as exc:
   record({'event':'failed','job':job,'error_type':type(exc).__name__,'message':str(exc)})
   raise
 return {'status':'passed','jobs':9}
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
 if a.run_id!='001':raise ValueError('unplanned run')
 s=importlib.util.spec_from_file_location('_pixel_nvml_scheduler_gate',D/'gate.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g);approval=g.admit_gpu()
 # Future gate must verify outer supervisor, input/weight identities, budget,
 # noGPU actual container and independent review before this continuation.
 plan=approval['plan'];jobs=approval['jobs'];out=D/'run_001';out.mkdir(exist_ok=False)
 if time.time()+plan['outer_timeout_seconds']+plan['grace_seconds']>=plan['budget_end_unix']:raise ValueError('remaining absolute budget')
 def run_one(job):
  i=job['order'];monitor=out/('monitor_job_%03d'%i)
  cmd=[PY,str(B/'monitor/supervise.py'),'--run-dir',str(monitor),'--timeout',str(plan['job_timeout_seconds']),'--snapshot-seconds','3600','--poll-seconds','5','--grace-seconds',str(plan['job_grace_seconds']),'--',PY,str(D/'controller_entry.py'),'--job-index',str(i)]
  result=subprocess.run(cmd,cwd=B,timeout=plan['job_timeout_seconds']+plan['job_grace_seconds']+30)
  terminal=json.loads((monitor/'terminal.json').read_text())
  if result.returncode!=0 or terminal.get('reason')!='completed' or terminal.get('worker_returncode')!=0 or terminal.get('remaining_process_group')!=[]:raise ValueError('job supervisor failed')
  return json.loads((out/('job_%03d'%i)/'memory_summary.json').read_text())
 result=run_all(jobs,run_one,lambda event:append(out/'journal.jsonl',event))
 if g.admit_gpu()!=approval:raise ValueError('source/weight admission changed')
 with (out/'final.json').open('x') as f:json.dump(result,f);f.write('\n');f.flush();os.fsync(f.fileno())
 return 0
if __name__=='__main__':raise SystemExit(main())
