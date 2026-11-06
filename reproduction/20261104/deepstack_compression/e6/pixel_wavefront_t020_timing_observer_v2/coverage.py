"""Supplemental per-job watcher windows; never delete or reweight timing rows."""
import argparse,hashlib,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review';R=B/'e6/pixel_wavefront_t020_timing_final_v9';A=B/'e6/pixel_wavefront_t020_timing_accept_v5'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');c=s['command'];cwd=Path(s['cwd'])
 if (len(c)<2 or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or (cwd/c[1]).resolve()!=script.resolve() or (out is not None and (c.count('--out')!=1 or (cwd/c[c.index('--out')+1]).resolve()!=out.resolve()))):raise ValueError('supervisor')
 return s,t
def classify(jobs,records,events):
 if len(jobs)!=24 or len(records)!=24 or not events or events[-1].get('event')!='observer_end':raise ValueError('incomplete inventory')
 scans=[e for e in events if e.get('event')=='snapshot'];logs=[e for e in events if e.get('event')=='logging_complete']
 if not scans or len(scans)!=len(logs) or [e.get('index') for e in scans]!=list(range(len(scans))) or [e.get('index') for e in logs]!=list(range(len(scans))):raise ValueError('observer samples')
 ns=events[-1].get('time_namespace');end=events[-1].get('end_ns')
 if not isinstance(ns,str) or not ns or type(end) is not int or end<=0:raise ValueError('observer end time')
 if [j.get('order') for j in jobs]!=list(range(24)) or any(type(r.get('start_monotonic_ns')) is not int or type(r.get('first_token_monotonic_ns')) is not int or not 0<r['start_monotonic_ns']<r['first_token_monotonic_ns'] for r in records):raise ValueError('job ordering or time')
 for s,l in zip(scans,logs):
  if s.get('time_namespace')!=ns or l.get('time_namespace')!=ns or any(type(s[k].get(x)) is not int for k in ('ps','docker') for x in ('query_start_ns','query_end_ns')):raise ValueError('clock namespace/window')
  if not (0<s['ps']['query_start_ns']<=s['ps']['query_end_ns']<=s['docker']['query_start_ns']<=s['docker']['query_end_ns']<=l.get('first_fsync_end_ns',0)<=end):raise ValueError('query/fsync ordering')
 allsame=all(r.get('boundaries',{}).get('time_namespace')==ns for r in records);start=min(r['start_monotonic_ns'] for r in records);finish=max(r['first_token_monotonic_ns'] for r in records)
 uptime='covered' if allsame and scans[0]['ps']['query_start_ns']<=start and end>=finish else 'unknown_or_partial'
 out=[]
 for job,record in zip(jobs,records):
  a,b=record['start_monotonic_ns'],record['first_token_monotonic_ns'];same=record.get('boundaries',{}).get('time_namespace')==ns
  def overlap(x,y):return x<=b and y>=a
  result={}
  for kind in ('ps','docker'):
   result[kind]={'status':'observed' if same and all(s[kind].get('returncode')==0 for s in scans) else 'unknown','overlap_indices':[s['index'] for s in scans if same and overlap(s[kind]['query_start_ns'],s[kind]['query_end_ns'])]}
  result['first_fsync_window_overlap_indices']=[s['index'] for s,l in zip(scans,logs) if same and overlap(s['ps']['query_start_ns'],l['first_fsync_end_ns'])]
  out.append({'order':job['order'],'windows':result})
 return {'status':'passed','observer_uptime':uptime,'jobs':24,'samples':len(scans),'annotations':out,'limitations':['60-second snapshots do not cover all background processes','second fsync duration and residual host effects unknown','observer queries can perturb timing; no samples removed']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if out!=(D/'coverage001.json').resolve():raise ValueError('unplanned output')
 # This supplemental result is only allowed after both formal and independent exits.
 formal=A/'verification001.json';review=I/'pixel_wavefront_t020_timing_result_review.json';reviewer=I/'review_pixel_wavefront_t020_timing_result.py';terminal(A/'monitor_verify001',A/'verify_entry.py',formal);terminal(I/'monitor_pixel_wavefront_t020_timing_result',reviewer,review);terminal(R/'monitor_run001',R/'launch.py')
 q=read(review)
 fields={'formal_sha256':formal,'formal_started_sha256':A/'monitor_verify001/started.json','formal_terminal_sha256':A/'monitor_verify001/terminal.json','run_started_sha256':R/'monitor_run001/started.json','run_terminal_sha256':R/'monitor_run001/terminal.json','reviewer_sha256':reviewer}
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('jobs')!=24 or q.get('GT_opened') is not False or q.get('errors')!=[] or any(q.get(k)!=sha(v) for k,v in fields.items()):raise ValueError('result review')
 start,term=terminal(D/'monitor_watch001',D/'observe.py');plan=read(D/'plan001.json')
 # Exact worker command checked by path/arguments, independent of interpreter alias.
 c=start['command'];cwd=Path(start['cwd']);expected_args=['--output',str(D/'observations001.jsonl'),'--stop',str(D/'stop001'),'--interval','60','--duration','30000']
 if Path(c[0]).resolve()!=Path('/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python').resolve() or c[2:]!=expected_args or plan['gpu_plan_sha256']!=sha(R/'gpu_plan001.json') or plan['formal_plan_sha256']!=sha(A/'verification_plan001.json') or any(sha(k)!=v for k,v in plan['source_sha256'].items()):raise ValueError('observer source or command')
 jobs=read(R/'gpu_plan001.json')['jobs'];records=[read(R/'run_001'/('job_%03d'%j['order'])/'record.json')['record'] for j in jobs];events=[json.loads(line) for line in (D/'observations001.jsonl').read_text().splitlines() if line.strip()];value=classify(jobs,records,events);value.update(formal_sha256=sha(formal),independent_review_sha256=sha(review),observer_sha256=sha(D/'observations001.jsonl'),GT_opened=False)
 with out.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
