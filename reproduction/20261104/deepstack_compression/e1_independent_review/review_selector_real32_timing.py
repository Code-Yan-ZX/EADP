"""Independently reconstruct the per-image real32 exploratory timing summaries."""
import argparse,json,math,statistics,sys
from pathlib import Path
B=Path(__file__).resolve().parent.parent;R=B/'e1_independent_review';S=B/'e5/selector_real32';D=B/'e5/selector_real32_failure';sys.path.insert(0,str(B/'e4_c_latency'));from identity_gate import sha
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert a.out.resolve()==R/'selector_real32_timing_review.json'
def read(p):return json.loads(p.read_text())
def terminal(mon,script):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in s['command'] if Path(x).name==script.name]==[script.resolve()];return s
terminal(D/'monitor_timing_summary001',D/'run_timing_summary.py');terminal(R/'monitor_selector_real32_failure',R/'review_selector_real32_failure.py');terminal(R/'monitor_selector_real32_run001',R/'review_selector_real32_run001.py');summary=read(D/'timing_summary_001.json');failure=read(R/'selector_real32_failure_review.json');assert summary['status']=='completed' and failure['status']=='passed' and summary['source_sha256']==sha(D/'timing_summary.py') and summary['runner_sha256']==sha(D/'run_timing_summary.py') and summary['journal_sha256']==sha(S/'run_001/journal.jsonl') and summary['failure_review_sha256']==sha(R/'selector_real32_failure_review.json')
events=[json.loads(x) for x in (S/'run_001/journal.jsonl').read_text().splitlines() if x.strip()];cases={};setup={}
for e in events:
 if e.get('status')=='graph_setup':assert e['budget'] not in setup;setup[e['budget']]=e['result']
 if e.get('status')!='case_completed':continue
 key=(e['image_index'],e['budget']);assert key not in cases and e['result']['status']=='measured';c=e['result'];d={}
 for method in ('author','fused','graph'):
  entries=[x for x in c['measurements'] if x['implementation']==method];assert len(entries)==5 and sorted(x['repeat'] for x in entries)==list(range(5));valid=method=='author' or c['candidate_matches_author'][method] is True
  if valid:
   assert all(x['status']=='measured' and x['equals_author'] is True and math.isfinite(x['gpu_event_ms']) and x['gpu_event_ms']>0 for x in entries);d[method]=statistics.median(x['gpu_event_ms'] for x in entries)
  else:assert all(x['status']=='unavailable' for x in entries);d[method]=None
 cases[key]=d
assert set(cases)=={(i,k) for i in range(32) for k in (128,256,512)} and set(setup)=={128,256,512}
for k in (128,256,512):
 actual=summary['budget_stats'][str(k)];assert actual['graph_setup']==setup[k] and setup[k]['input_copied_each_replay'] is True;paired=[i for i in range(32) if all(cases[i,k][m] is not None for m in ('author','fused','graph'))];assert actual['paired_image_indices']==paired and actual['excluded_image_indices']==[i for i in range(32) if i not in paired]
 for i in range(32):assert actual['per_image'][str(i)]==cases[i,k]
 for m in ('author','fused','graph'):
  for name,indices in [('all_available',range(32)),('paired',paired)]:
   values=[cases[i,k][m] for i in indices if cases[i,k][m] is not None];assert actual[name][m]=={'n':len(values),'median_ms':statistics.median(values),'min_ms':min(values),'max_ms':max(values)}
assert summary['budget_stats']['512']['excluded_image_indices']==[17]
r={'status':'passed','independent_review':True,'images':32,'budgets':3,'paired_counts':{str(k):len(summary['budget_stats'][str(k)]['paired_image_indices']) for k in (128,256,512)},'summary_sha256':sha(D/'timing_summary_001.json'),'summary_started_sha256':sha(D/'monitor_timing_summary001/started.json'),'summary_terminal_sha256':sha(D/'monitor_timing_summary001/terminal.json'),'reviewer_sha256':sha(__file__),'GT_opened':False,'scoring_performed':False,'limitations':['32-image post-pilot exploratory sample; K512 excludes one failed identity and 31 paired cases may be selected','graph is replay of fused kernel with copies; capture/setup excluded and allocated memory separate','component CUDA event timings are not end-to-end TTFT'], 'errors':[]};a.out.write_text(json.dumps(r,indent=2)+'\n');print('passed')
