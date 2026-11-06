"""Eight bounded process continuations; caller must already pass GPU gate."""
import json,os,time
from pathlib import Path
ARMS=(('author','wavefront'),('wavefront','author'),('wavefront','author'),('author','wavefront'))
def append(path,item):
 with Path(path).open('a') as f:json.dump(item,f,separators=(',',':'));f.write('\n');f.flush();os.fsync(f.fileno())
def validate_jobs(jobs):
 if len(jobs)!=8 or [x.get('order') for x in jobs]!=list(range(8)):raise ValueError('eight processes required')
 for i,j in enumerate(jobs):
  if j.get('block')!=i//2 or j.get('arm')!=ARMS[i//2][i%2] or len(j.get('measured',[]))!=48:raise ValueError('ABBA process schedule')
  if i%2 and j['measured']!=jobs[i-1]['measured']:raise ValueError('same block order required')
 return True
def run(jobs,out,run_job,check_job,check_supervisor,budget_end,inner,cleanup,now=time.time):
 validate_jobs(jobs);check_supervisor()
 if inner<=0 or cleanup<=0 or now()+len(jobs)*inner+cleanup>=budget_end:raise ValueError('insufficient time budget')
 out=Path(out);out.mkdir(exist_ok=False);journal=out/'controller.jsonl';seen=set();append(journal,{'event':'controller_started','processes':8})
 try:
  for job in jobs:
   check_supervisor()
   if now()+(8-job['order'])*inner+cleanup>=budget_end:raise ValueError('remaining time budget')
   append(journal,{'event':'job_started','order':job['order'],'arm':job['arm'],'block':job['block']});d=out/('job_%03d'%job['order']);item=run_job(job,d);ident=item.get('container_identity') or {};host=ident.get('host_init') or {};cid=ident.get('container_id');process=(host.get('pid'),host.get('start_ticks'),host.get('boot_id'))
   if not isinstance(cid,str) or len(cid)!=64 or type(process[0]) is not int or process[0]<=0 or type(process[1]) is not int or process[1]<=0 or not process[2] or cid in seen or process in seen:raise ValueError('container/host process identity')
   check_job(job,d,item);seen.update((cid,process));append(journal,{'event':'job_completed','order':job['order'],'arm':job['arm'],'block':job['block'],'container_id':cid,'host_init':host})
  append(journal,{'event':'all_completed','processes':8,'warmups':8,'measured':384})
 except BaseException as exc:
  append(journal,{'event':'controller_failed','error_type':type(exc).__name__,'message':str(exc)});raise
 return {'status':'completed','processes':8,'warmups':8,'measured':384}
