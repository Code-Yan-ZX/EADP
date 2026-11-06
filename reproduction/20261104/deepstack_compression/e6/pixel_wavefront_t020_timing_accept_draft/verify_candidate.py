"""CPU-only formal verification candidate; not frozen and no executable entry."""
import json,hashlib
from pathlib import Path

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate(jobs, references, run_dir, job_check):
 """Recheck every frozen job, actual files, raw output and container exit.
 Call only after separate provenance/supervisor/Docker admission and exit checks.
 """
 root=Path(run_dir);lines=[json.loads(x) for x in (root/'controller.jsonl').read_text().splitlines() if x.strip()]
 if len(jobs)!=24 or len(lines)!=50 or lines[0].get('event')!='controller_started' or lines[-1]!={'event':'all_completed','jobs':24,'warmups':6,'measured':18}:raise ValueError('controller inventory')
 files={};containers=set();processes=set();rows=[]
 for i,job in enumerate(jobs):
  if job.get('order')!=i or lines[2*i+1]!={'event':'job_started','job':job}:raise ValueError('job start order')
  event=lines[2*i+2];d=root/('job_%03d'%i)
  if event.get('event')!='job_complete' or event.get('job')!=job:raise ValueError('job completion')
  launch=json.loads((d/'launch.json').read_text());release=json.loads((d/'release.json').read_text());exit=json.loads((d/'exit.json').read_text());stored=json.loads((d/'record.json').read_text())
  ident=exit.get('container_identity') or {};host=ident.get('host_init') or {};cid=ident.get('container_id');process=(host.get('pid'),host.get('start_ticks'),host.get('boot_id'))
  if (launch.get('job')!=job or release.get('job')!=job or stored.get('job')!=job or release.get('launch_sha256')!=digest(d/'launch.json')
      or not isinstance(cid,str) or len(cid)!=64 or cid in containers or process in processes or event.get('container_id')!=cid or event.get('host_init')!=host
      or release.get('container_id')!=cid or release.get('host_init')!=host or release.get('controller')!=launch.get('controller')
      or type(process[0]) is not int or process[0]<=0 or type(process[1]) is not int or process[1]<=0 or not process[2]
      or exit.get('returncode')!=0 or exit.get('container_exit_code')!=0 or exit.get('container_stopped_verified') is not True
      or exit.get('gpu_pids_after')!=[] or exit.get('timed_out') is not False or exit.get('interrupted') is not None
      or exit.get('client_exited_while_container_running') is not False or exit.get('failure_type') is not None):raise ValueError('container, release or completion evidence')
  containers.add(cid);processes.add(process);key=(job['task'],str(job['qid']),job['arm']);metrics=job_check(job,stored['record'],references[key],d/'raw.json',d)
  if metrics!=stored.get('metrics'):raise ValueError('stored metrics mismatch')
  for name in ('launch.json','release.json','exit.json','record.json','raw.json'):
   p=d/name;files[str(p.resolve())]=digest(p)
  rows.append({'order':i,'phase':job['phase'],'ttft_ns':metrics['ttft_ns']})
 if sum(j['phase']=='warmup' for j in jobs)!=6 or sum(j['phase']=='measurement' for j in jobs)!=18:raise ValueError('phase counts')
 return {'status':'passed','jobs':24,'warmups':6,'measured':18,'files_sha256':files,'rows':rows,'GT_opened':False,'scoring_performed':False,'errors':[], 'limitations':['actual Docker inspections, source snapshot, supervisor exit and background windows require separate evidence','three-anchor exploratory instrumented path','separate processes mean warmup is not in-process steady state']}
if __name__=='__main__':raise SystemExit('not frozen or admitted; candidate only')
