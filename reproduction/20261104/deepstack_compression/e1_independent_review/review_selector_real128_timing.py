"""Recompute image-level timing from both accepted journals, no GT/GPU."""
import argparse,hashlib,json,math,os,statistics
from pathlib import Path
B=Path(__file__).resolve().parent.parent;I=B/'e1_independent_review';A=B/'e5/selector_real128_analysis'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def term(m,s):
 x=read(m/'started.json');t=read(m/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and not t['remaining_process_group'] and x['worker_pid']==t['worker_pid'] and any((Path(x['cwd'])/z).resolve()==s.resolve() for z in x['command'])
def dist(v):return {'n':len(v),'median_ms':statistics.median(v),'min_ms':min(v),'max_ms':max(v)} if v else {'n':0}
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert a.out.resolve()==I/'selector_real128_timing_review.json'
term(A/'monitor_timing001',A/'timing_run.py');term(I/'monitor_selector_real128_combined',I/'review_selector_real128_combined.py');combined=read(A/'combined001.json');review=read(I/'selector_real128_combined_review.json');timing=read(A/'timing001.json');assert review['status']=='passed' and review['combined_sha256']==timing['combined_sha256']==sha(A/'combined001.json') and timing['source_sha256']==sha(A/'timing_run.py') and timing['GT_opened'] is False and timing['scoring_performed'] is False
allvals={str(k):{} for k in (128,256,512)};setups={str(k):{} for k in (128,256,512)}
for label,folder,count,field in [('observed_pilot',B/'e5/selector_real32',32,'prior_evidence'),('added',B/'e5/selector_real128',96,'expansion_evidence')]:
 journal=folder/'run_001/journal.jsonl';assert sha(journal)==combined[field]['journal_sha256'];events=[json.loads(x) for x in journal.read_text().splitlines() if x.strip()]
 for e in events:
  if e.get('status')=='graph_setup':
   k=str(e['budget']);s=e['result'];assert label not in setups[k] and s['status']=='passed';before=s['before_bytes'];after=s['after_bytes'];setups[k][label]={'capture_wall_ms':s['capture_wall_ms'],'allocated_before_bytes':before['allocated'],'allocated_after_bytes':after['allocated'],'allocated_delta_bytes':after['allocated']-before['allocated'],'reserved_before_bytes':before['reserved'],'reserved_after_bytes':after['reserved']}
  if e.get('status')!='case_completed':continue
  i=e['image_index'];k=str(e['budget']);key=f'{label}:{i}';assert 0<=i<count and k in allvals and key not in allvals[k];v=e['result'];assert v['status']=='measured';row={}
  for method in ('author','fused','graph'):
   ms=[x for x in v['measurements'] if x['implementation']==method];assert len(ms)==5 and sorted(x['repeat'] for x in ms)==list(range(5));allowed=method=='author' or v['candidate_matches_author'][method] is True
   if not allowed:assert all(x['status']=='unavailable' for x in ms);row[method]=None;continue
   assert all(x['status']=='measured' and x['equals_author'] is True and type(x['gpu_event_ms']) in (int,float) and math.isfinite(x['gpu_event_ms']) and x['gpu_event_ms']>0 for x in ms);row[method]=statistics.median(x['gpu_event_ms'] for x in ms)
  allvals[k][key]=row
results={}
for k,per in allvals.items():
 assert len(per)==128 and per==timing['results'][k]['per_image_medians_ms'] and setups[k]==timing['results'][k]['graph_setup'];missing=[i for i,v in per.items() if any(v[m] is None for m in ('author','fused','graph'))];paired=[v for v in per.values() if all(v[m] is not None for m in ('author','fused','graph'))];assert missing==timing['results'][k]['excluded_from_paired'];assert len(paired)==(127 if k=='512' else 128)
 for m in ('author','fused','graph'):
  assert dist([v[m] for v in per.values() if v[m] is not None])==timing['results'][k]['all_available'][m]
  assert dist([v[m] for v in paired])==timing['results'][k]['paired'][m]
 results[k]={'paired':timing['results'][k]['paired'],'excluded':missing,'graph_setup':setups[k]}
out={'status':'passed','independent_review':True,'images':128,'results':results,'timing_sha256':sha(A/'timing001.json'),'timing_started_sha256':sha(A/'monitor_timing001/started.json'),'timing_terminal_sha256':sha(A/'monitor_timing001/terminal.json'),'combined_review_sha256':sha(I/'selector_real128_combined_review.json'),'reviewer_sha256':sha(__file__),'GT_opened':False,'scoring_performed':False,'limitations':['K512 paired analysis excludes the single strict identity failure and has selection bias','pilot observed before expansion; not an independent 128-image confirmation','light code editing and other background activity unquantified; no per-sample windows','component CUDA event timing, not TTFT; graph setup separate','allocator figures do not represent device-wide peak'],'errors':[]}
with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print('passed')
