"""Annotate observed ps/query and first-fsync windows; no TTFT values reported."""
import argparse,hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;RUN=BASE/'e4_c_budget_ttft_pilot_v3';ENV=BASE/'e4_c_budget_ttft_environment'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(start,end):
 s=read(start);t=read(end)
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid'):raise ValueError('supervisor incomplete')
def windows(rows,namespace):
 if not rows or len(rows)%2:raise ValueError('missing watcher pair')
 out=[]
 for i in range(0,len(rows),2):
  q,l=rows[i:i+2];start=q.get('query_start_ns');query_end=q.get('query_end_ns');log_end=l.get('logging_end_ns')
  if (q.get('event')!='snapshot' or l.get('event')!='snapshot_logging_complete' or l.get('query_start_ns')!=start or q.get('time_namespace')!=namespace or
      any(type(x) is not int for x in (start,query_end,log_end)) or not start<=query_end<=log_end):raise ValueError('invalid watcher window or namespace')
  out.append({'query_window':[start,query_end],'query_and_first_fsync_window':[start,log_end],'query_returncode':q.get('returncode'),'query_error':q.get('error_type')})
 return out
def matches(jobs,events,observed,namespace):
 if len(jobs)!=48 or len(events)!=144:raise ValueError('incomplete timing journal')
 out=[]
 for i,job in enumerate(jobs):
  e=events[i*3:(i+1)*3]
  if [x.get('event') for x in e]!=['started','raw_generation','completed'] or any(x.get('job')!=job for x in e):raise ValueError('timing journal order')
  b=e[2]['payload']['boundaries'];start=b['t0_before_image_read_ns'];first=b['first_token_selected_ns'];end=b['generation_complete_ns']
  if b.get('time_namespace')!=namespace or not start<first<=end:raise ValueError('timing clock/namespace mismatch')
  def overlap(a,b):return a[0]<=b[1] and b[0]<=a[1]
  out.append({'order':i,'phase':job['phase'],'task':job['task'],'budget':job['budget'],
   'ttft_ps_query_overlap':[j for j,w in enumerate(observed) if overlap([start,first],w['query_window'])],
   'ttft_ps_first_fsync_overlap':[j for j,w in enumerate(observed) if overlap([start,first],w['query_and_first_fsync_window'])],
   'full_job_ps_first_fsync_overlap':[j for j,w in enumerate(observed) if overlap([start,end],w['query_and_first_fsync_window'])],
   'second_fsync_overlap':'unknown'})
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.resolve()!=HERE/'background_overlap001.json':raise ValueError('unplanned output')
 plan=read(HERE/'background_plan001.json')
 for path,digest in plan['source_sha256'].items():
  if sha(path)!=digest:raise ValueError('background source changed')
 formal=HERE/'verification001.json';fs=HERE/'monitor_verify001/started.json';ft=HERE/'monitor_verify001/terminal.json';terminal(fs,ft);f=read(formal)
 if f.get('status')!='passed' or f.get('jobs')!=48 or f.get('errors')!=[]:raise ValueError('formal timing acceptance missing')
 terminal(ENV/'monitor_watch001/started.json',ENV/'monitor_watch001/terminal.json')
 journal=RUN/'run_001/journal.jsonl';launch=read(RUN/'run_001/launch.json');health=RUN/'monitor_run001/health.jsonl'
 if f.get('journal_sha256')!=sha(journal) or f.get('health_sha256')!=sha(health):raise ValueError('timing evidence changed')
 records=[json.loads(x) for x in (ENV/'process_samples001.jsonl').read_text().splitlines() if x.strip()];observed=windows(records,launch['time_namespace'])
 jobs=read(RUN/'pilot_plan001.json')['jobs'];events=[json.loads(x) for x in journal.read_text().splitlines() if x.strip()];annotations=matches(jobs,events,observed,launch['time_namespace'])
 sources=[ENV/'watch.py',ENV/'process_samples001.jsonl',ENV/'after001.json',ENV/'stop001',ENV/'monitor_watch001/started.json',ENV/'monitor_watch001/terminal.json']
 result={'status':'passed','jobs':48,'watcher_windows':observed,'annotations':annotations,'GT_opened':False,'scoring_performed':False,
  'formal_sha256':sha(formal),'formal_started_sha256':sha(fs),'formal_terminal_sha256':sha(ft),'journal_sha256':sha(journal),'watcher_source_sha256':{str(x):sha(x) for x in sources},
  'limits':'Intervals cover ps query and first fsync only; second log write/fsync and residual/background activity are not bounded. No samples excluded; ps CPU is process-lifetime percentage.',
  'source_sha256':sha(__file__),'plan_sha256':sha(HERE/'background_plan001.json'),'errors':[]}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
