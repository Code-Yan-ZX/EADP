"""Independent reconstruction of 32+96 selector case accounting; no GT."""
import argparse,hashlib,json,os
from collections import Counter
from pathlib import Path
B=Path(__file__).resolve().parent.parent; A=B/'e5/selector_real128_analysis'; I=B/'e1_independent_review'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def terminal(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and not t['remaining_process_group'] and s['worker_pid']==t['worker_pid']; cmd=s['command'];cwd=Path(s['cwd']);assert any((cwd/x).resolve()==script.resolve() for x in cmd)
 if out is not None:assert '--out' in cmd and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
 return s,t
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();assert args.out.resolve()==I/'selector_real128_combined_review.json'
result_path=A/'combined001.json'; result=read(result_path);terminal(A/'monitor_combine001',A/'combine_run.py',result_path)
assert result['status']=='completed' and result['GT_opened'] is False and result['scoring_performed'] is False and result['reviewer_sha256']==sha(A/'combine_run.py')
P=B/'e5/selector_real32';N=B/'e5/selector_real128';manifest=P/'sample_manifest_001.json';m=read(manifest);assert sha(manifest)==result['manifest_sha256'];old=m['pilot'];full=m['full_128']
def key(x):return (x['task'],str(x['qid']),x['group_id'])
assert len(old)==32 and len(full)==128 and len({key(x) for x in full})==128 and len({x['group_id'] for x in full})==128
oldkeys={key(x) for x in old};assert len(oldkeys)==32 and oldkeys<={key(x) for x in full};new=[x for x in full if key(x) not in oldkeys];assert len(new)==96
summaries={};allfail=[]
for folder,rows,label,n,proof,reviewer,reviewmon,evidence in [(P,old,'observed_pilot_32',96,'selector_real32_run001_review.json','review_selector_real32_run001.py','monitor_selector_real32_run001','prior_evidence'),(N,new,'added_96',288,'selector_real128_run001_review.json','review_selector_real128_run001.py','monitor_selector_real128_run001','expansion_evidence')]:
 fpath=folder/'run_001_verification.json';rp=I/proof;f=read(fpath);r=read(rp);terminal(folder/'monitor_verify_run001',folder/'verify.py',fpath);terminal(I/reviewmon,I/reviewer,rp);terminal(folder/'monitor_run001',folder/'launch.py');assert f['status']=='passed' and r['status']=='passed' and r['independent_review'] is True and f['cases']==r['cases']==n and f['images']==r['images']==len(rows) and f['GT_opened'] is r['GT_opened'] is False and f['scoring_performed'] is r['scoring_performed'] is False and r['formal_sha256']==sha(fpath) and r['reviewer_sha256']==sha(I/reviewer) and r['array_sha256']==f['array_sha256']
 for field,path in [('formal_started_sha256',folder/'monitor_verify_run001/started.json'),('formal_terminal_sha256',folder/'monitor_verify_run001/terminal.json'),('run_started_sha256',folder/'monitor_run001/started.json'),('run_terminal_sha256',folder/'monitor_run001/terminal.json')]:assert r[field]==sha(path)
 journal=folder/'run_001/journal.jsonl';jsha=sha(journal);assert f['files_sha256'][str(journal.resolve())]==jsha and result[evidence]=={'formal_sha256':sha(fpath),'review_sha256':sha(rp),'journal_sha256':jsha,'review_terminal_sha256':sha(I/reviewmon/'terminal.json')}
 cases={};events=[json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
 for e in events:
  if e.get('status')!='case_completed':continue
  index=e['image_index'];k=e['budget'];assert 0<=index<len(rows) and k in (128,256,512) and (index,k) not in cases;cases[(index,k)]=e['result']
 assert len(cases)==n and set(cases)=={(i,k) for i in range(len(rows)) for k in (128,256,512)}
 groups={}
 for k in (128,256,512):
  groups[str(k)]={}
  for method in ('fused','graph'):
   c=Counter()
   for i,row in enumerate(rows):
    x=cases[(i,k)];ident=x.get('identity',{}) if isinstance(x,dict) else {};cand=ident.get(method+'_ordered_ids');measure=[z for z in x.get('measurements',[]) if z['implementation']==method] if isinstance(x,dict) else []
    if not isinstance(x,dict) or x.get('status')!='measured' or cand is None:c['unavailable']+=1;continue
    author=x['author_endpoint']['ordered_ids'];assert len(author)==len(cand)==k and len(set(author))==len(set(cand))==k
    ordered=author==cand;sets=set(author)==set(cand);assert ordered==ident[method+'_equals_author'];assert len(measure)==5
    repeats=all(z['status']=='measured' and z.get('ordered_ids')==author and z.get('equals_author') is True for z in measure)
    assert repeats is (x['candidate_matches_author'][method] is True)
    c['strict_order_pass' if ordered and repeats else 'strict_order_fail']+=1;c['initial_spatial_equal_posthoc' if sets else 'initial_spatial_different_posthoc']+=1
    if not ordered:
     assert all(z['status']=='unavailable' and 'gpu_event_ms' not in z and 'host_wall_ms' not in z for z in measure);allfail.append({'task':row['task'],'qid':str(row['qid']),'budget':k,'method':method,'same_set':sets,'first_difference_zero_based':next(j for j,(a,b) in enumerate(zip(author,cand)) if a!=b)})
   groups[str(k)][method]={'planned':len(rows),**{q:c[q] for q in ('strict_order_pass','strict_order_fail','unavailable','initial_spatial_equal_posthoc','initial_spatial_different_posthoc')}}
 assert groups==result['counts'][label];summaries[label]=groups
combined={}
for k in ('128','256','512'):
 combined[k]={}
 for method in ('fused','graph'):
  left=summaries['observed_pilot_32'][k][method];right=summaries['added_96'][k][method];combined[k][method]={field:left[field]+right[field] for field in left}
assert combined==result['counts']['combined_128'] and len(allfail)==2 and all(x['task']=='TextVQA_VAL' and x['qid']=='35191' and x['budget']==512 and x['same_set'] and x['first_difference_zero_based']==332 for x in allfail)
out={'status':'passed','independent_review':True,'images':128,'planned_cases':384,'case_counts':{'pilot':96,'added':288},'counts':combined,'failures':allfail,'failure_candidate_measurements_unavailable':True,'combined_sha256':sha(result_path),'combined_started_sha256':sha(A/'monitor_combine001/started.json'),'combined_terminal_sha256':sha(A/'monitor_combine001/terminal.json'),'reviewer_sha256':sha(__file__),'GT_opened':False,'scoring_performed':False,'limitations':['post-pilot expansion is exploratory, not an independent 128-image confirmation','spatial set comparison is posthoc and cannot grant strict ordered equivalence','fused and graph share implementation; graph is not an independent algorithm replication','GPU round scores were not archived'],'errors':[]}
with args.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print('passed')
