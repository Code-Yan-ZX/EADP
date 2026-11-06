"""Independent post-hoc paired-score/bootstrap check; no new labels or predictions."""
import collections,hashlib,json,math
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];H=R/'e6/selected_vs_eadp';I=R/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def mon(p):
 s=read(p/'started.json');t=read(p/'terminal.json');assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid'];return s
result=H/'result_001.json';plan=H/'plan_003.json';proof=H/'independent_prelaunch_review_002.json'
p=read(plan);r=read(result);rv=read(proof);start=mon(H/'monitor_compare002');mon(I/'monitor_selected_vs_eadp_prelaunch002')
assert any(Path(x).name=='compare.py' for x in start['command']) and r['status']=='completed' and r['plan_sha256']==sha(plan) and r['source_sha256']==sha(H/'compare.py')
assert rv['status']=='passed' and rv['independent_review'] is True and rv['reviewer_sha256']==sha(I/'review_selected_vs_eadp002.py') and all(sha(path)==h for path,h in rv['source_sha256'].items())
assert all(sha(path)==h for path,h in p['inputs_sha256'].items()) and np.__version__==p['numpy_version']
def key(x):return (x['task'],str(x['qid']),x['group_id'])
def unique(xs):
 d={}
 for x in xs:
  k=key(x);assert k not in d;d[k]=x
 return d
a=unique(lines(R/'e1/e1full001_score/scores.jsonl'));b=unique(lines(R/'e6/selected_analysis/score002/scores.jsonl'));assert len(a)==len(b)==384 and set(a)==set(b)
paired=[]
for k,x in a.items():
 e=x['arms']['P3-D1'];z=b[k]['arms'];vals=[e['score'],z['div:adapted']['score'],z['cd:adapted']['score']]
 for one in (e,z['div:adapted'],z['cd:adapted']):assert math.isfinite(one['score']) and math.isfinite(one['independent_score']) and abs(one['score']-one['independent_score'])<=1e-12
 assert all(0<=v<=1 for v in vals);paired.append((k[0],k[2],vals))
for task in p['strata']:
 rows=[x for x in paired if task=='ALL' or x[0]==task];groups=collections.defaultdict(list)
 for _,g,v in rows:groups[g].append(v)
 names=sorted(groups);n=len(names);rng=np.random.default_rng(p['bootstrap']['seed']);draws=rng.integers(0,n,size=(p['bootstrap']['replicates'],n));gs=np.array([np.array(groups[g]).sum(axis=0) for g in names]);sizes=np.array([len(groups[g]) for g in names]);bootstrap=gs[draws].sum(axis=1)/sizes[draws].sum(axis=1)[:,None];actual=r['reports'][task]
 assert actual['n_questions']==len(rows) and actual['n_groups']==n
 for idx,name in ((1,'div-adapted_minus_eadp'),(2,'cd-adapted_minus_eadp')):
  d=np.array([v[idx]-v[0] for _,_,v in rows]);ci=np.quantile(100*(bootstrap[:,idx]-bootstrap[:,0]),[.025,.975]);c=actual['contrasts'][name]
  assert abs(c['difference_pp']-float(100*d.mean()))<=1e-10 and np.allclose(c['percentile_95_ci_pp'],ci,rtol=0,atol=1e-10)
  assert (c['improved'],c['declined'],c['unchanged'])==(int((d>1e-12).sum()),int((d< -1e-12).sum()),int((abs(d)<=1e-12).sum()))
out=I/'selected_vs_eadp_result_review.json';q={'status':'passed','independent_review':True,'n_questions':384,'tasks_verified':len(p['strata']),'contrasts_verified':2*len(p['strata']),'result_sha256':sha(result),'plan_sha256':sha(plan),'compare_started_sha256':sha(H/'monitor_compare002/started.json'),'compare_terminal_sha256':sha(H/'monitor_compare002/terminal.json'),'reviewer_sha256':sha(__file__),'GT_opened':False,'new_predictions_opened':False,'errors':[],'limitations':p['constraints']}
with out.open('x') as f:json.dump(q,f,indent=2);f.write('\n')
