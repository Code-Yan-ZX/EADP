"""Posthoc counts for 32 observed + 96 additional; locked until both independent reviews."""
import argparse,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent.parent;R=BASE/'e1_independent_review';P=BASE/'e5/selector_real32';N=BASE/'e5/selector_real128';sys.path[:0]=[str(BASE/'e4_c_latency'),str(N)]
from identity_gate import sha,supervised
from latency_admission import terminal
from expansion_contract import load_manifest,selected
from combine_contract import summarize

def req(x,m):
 if not x:raise ValueError(m)
def gate(folder,name,n_images,n_cases,reviewer,monitor):
 formal=folder/'run_001_verification.json';review=R/name;fs=folder/'monitor_verify_run001/started.json';ft=folder/'monitor_verify_run001/terminal.json';rs=R/monitor/'started.json';rt=R/monitor/'terminal.json';run_s=folder/'monitor_run001/started.json';run_t=folder/'monitor_run001/terminal.json'
 supervised(fs,ft,folder/'verify.py',formal);supervised(rs,rt,R/reviewer,review);terminal(run_s,run_t);f=json.loads(formal.read_text());r=json.loads(review.read_text())
 req(f['status']=='passed' and r['status']=='passed' and r['independent_review'] is True and f['images']==r['images']==n_images and f['cases']==r['cases']==n_cases and f['GT_opened'] is False and r['GT_opened'] is False and r['formal_sha256']==sha(formal) and r['formal_started_sha256']==sha(fs) and r['formal_terminal_sha256']==sha(ft) and r['run_started_sha256']==sha(run_s) and r['run_terminal_sha256']==sha(run_t) and r['reviewer_sha256']==sha(R/reviewer) and f['array_sha256']==r['array_sha256'],'incomplete independent review')
 journal=folder/'run_001/journal.jsonl';req(f['files_sha256'].get(str(journal.resolve()))==sha(journal),'journal identity')
 return f,{'formal_sha256':sha(formal),'review_sha256':sha(review),'journal_sha256':sha(journal),'review_terminal_sha256':sha(rt)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();sample,remaining=load_manifest(P/'sample_manifest_001.json');old,oi=gate(P,'selector_real32_run001_review.json',32,96,'review_selector_real32_run001.py','monitor_selector_real32_run001');new,ni=gate(N,'selector_real128_run001_review.json',96,288,'review_selector_real128_run001.py','monitor_selector_real128_run001')
 cases=[]
 for line in (P/'run_001/journal.jsonl').read_text().splitlines():
  e=json.loads(line)
  if e.get('status')!='case_completed':continue
  idx=e['image_index'];row=sample['pilot'][idx];item={**row,'budget':e['budget']};v=e['result']
  for method in ('fused','graph'):
   if not isinstance(v,dict) or v.get('status')!='measured' or v.get('identity',{}).get(method+'_ordered_ids') is None:item[method]=None;continue
   c=selected(v['author_endpoint']['ordered_ids'],v['identity'][method+'_ordered_ids'],e['budget']);c['all_repeats_ordered_equal']=v['candidate_matches_author'][method] is True;item[method]=c
  cases.append(item)
 expanded=[]
 for e in new['outcomes']:
  row=remaining[e['image_index']];req(e['task']==row['task'] and str(e['qid'])==str(row['qid']),'expansion row mismatch');expanded.append({**row,'budget':e['budget'],'fused':e['fused'],'graph':e['graph']})
 result={'status':'completed','scope':'32 observed pilot plus 96 added DEV images; expansion chosen after pilot; not an independent 128-image validation','counts':summarize(sample['pilot'],remaining,cases,expanded),'prior_evidence':oi,'expansion_evidence':ni,'manifest_sha256':sha(P/'sample_manifest_001.json'),'GT_opened':False,'scoring_performed':False,'reviewer_sha256':sha(__file__),'limitations':['Ordered pass requires original strict full order and all repeated outputs; initial set equality is a separately reported posthoc diagnostic.','Unavailable and all failure cases remain in planned denominators.','Fused and fused graph share an implementation and are not independent algorithm replications.']}
 with a.out.open('x') as h:json.dump(result,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
if __name__=='__main__':main()
