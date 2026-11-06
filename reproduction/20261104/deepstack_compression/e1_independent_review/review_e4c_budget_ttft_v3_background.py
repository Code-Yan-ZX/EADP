"""Independently reconstruct watcher-window overlap; no inference or GPU."""
import argparse,hashlib,json,os
from pathlib import Path
B=Path(__file__).resolve().parents[1];A=B/'e4_c_budget_ttft_accept_v3';D=B/'e4_c_budget_ttft_pilot_v3';E=B/'e4_c_budget_ttft_environment';I=B/'e1_independent_review';OUT=I/'e4c_budget_ttft_v3_background_review.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def term(path,script=None):
 s=read(path/'started.json');t=read(path/'terminal.json');c=s['command']
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or s.get('worker_pid')!=s.get('process_group'):raise ValueError('unclean supervisor')
 if script and (Path(s['cwd'])/c[1]).resolve()!=script.resolve():raise ValueError('supervisor command')
def run():
 main=I/'e4c_budget_ttft_v3_result_review.json';m=read(main);term(I/'monitor_e4c_budget_ttft_v3_result',I/'review_e4c_budget_ttft_v3_result.py')
 if m.get('status')!='passed' or m.get('reviewer_sha256')!=sha(I/'review_e4c_budget_ttft_v3_result.py') or m.get('formal_sha256')!=sha(A/'verification001.json'):raise ValueError('independent timing review')
 p=A/'background_plan001.json';r=A/'background_overlap001.json';proof=read(r);plan=read(p);term(A/'monitor_background001',A/'background_overlap.py');term(E/'monitor_watch001',E/'watch.py')
 if proof.get('status')!='passed' or proof.get('jobs')!=48 or proof.get('errors')!=[] or proof.get('plan_sha256')!=sha(p) or proof.get('source_sha256')!=sha(A/'background_overlap.py') or proof.get('formal_sha256')!=sha(A/'verification001.json'):raise ValueError('background proof')
 for path,d in plan['source_sha256'].items():
  if sha(path)!=d:raise ValueError('background source mismatch')
 for path,d in proof['watcher_source_sha256'].items():
  if sha(path)!=d:raise ValueError('watcher identity')
 ns=read(D/'run_001/launch.json')['time_namespace'];rows=[json.loads(x) for x in (E/'process_samples001.jsonl').read_text().splitlines() if x.strip()]
 if len(rows)!=6:raise ValueError('watcher sample count')
 windows=[]
 for k in range(0,len(rows),2):
  q,l=rows[k:k+2];a=q.get('query_start_ns');b=q.get('query_end_ns');c=l.get('logging_end_ns')
  if (q.get('event')!='snapshot' or l.get('event')!='snapshot_logging_complete' or l.get('query_start_ns')!=a or q.get('time_namespace')!=ns or any(type(x) is not int for x in (a,b,c)) or not a<=b<=c):raise ValueError('invalid watcher timing')
  windows.append({'query_window':[a,b],'query_and_first_fsync_window':[a,c],'query_returncode':q.get('returncode'),'query_error':q.get('error_type')})
 events=[json.loads(x) for x in (D/'run_001/journal.jsonl').read_text().splitlines() if x.strip()];jobs=read(D/'pilot_plan001.json')['jobs'];annotations=[]
 if len(events)!=144 or len(jobs)!=48 or windows!=proof['watcher_windows']:raise ValueError('window or denominator')
 for i,job in enumerate(jobs):
  e=events[i*3:(i+1)*3]
  if [x.get('event') for x in e]!=['started','raw_generation','completed'] or any(x.get('job')!=job for x in e):raise ValueError('event mismatch')
  b=e[2]['payload']['boundaries'];a=b['t0_before_image_read_ns'];z=b['first_token_selected_ns'];end=b['generation_complete_ns']
  if b.get('time_namespace')!=ns or not a<z<=end:raise ValueError('timing namespace')
  def intersections(x,y,key):return [j for j,w in enumerate(windows) if x<=w[key][1] and w[key][0]<=y]
  annotations.append({'order':i,'phase':job['phase'],'task':job['task'],'budget':job['budget'],'ttft_ps_query_overlap':intersections(a,z,'query_window'),'ttft_ps_first_fsync_overlap':intersections(a,z,'query_and_first_fsync_window'),'full_job_ps_first_fsync_overlap':intersections(a,end,'query_and_first_fsync_window'),'second_fsync_overlap':'unknown'})
 if annotations!=proof['annotations']:raise ValueError('overlap annotation mismatch')
 return {'status':'passed','independent_review':True,'GT_opened':False,'scoring_performed':False,'jobs':48,'watcher_windows':3,'errors':[], 'background_overlap_sha256':sha(r),'background_started_sha256':sha(A/'monitor_background001/started.json'),'background_terminal_sha256':sha(A/'monitor_background001/terminal.json'),'main_review_sha256':sha(main),'reviewer_sha256':sha(__file__),'ttft_query_overlaps':sum(bool(x['ttft_ps_query_overlap']) for x in annotations),'ttft_query_plus_first_fsync_overlaps':sum(bool(x['ttft_ps_first_fsync_overlap']) for x in annotations),'limitations':['The second watcher fsync and residual or other background are not bounded; all samples retained.']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.resolve()!=OUT:raise ValueError('output')
 v=run()
 with a.out.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
