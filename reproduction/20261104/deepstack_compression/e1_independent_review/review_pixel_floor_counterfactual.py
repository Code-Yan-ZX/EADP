"""Independently rebuild post-hoc routing, K summaries and paired bootstrap."""
import argparse,collections,hashlib,json,math,os
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent.parent;D=B/'e6/pixelprune_floor_counterfactual';R=B/'e1_independent_review'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def key(x):return (x['task'],str(x['qid']),x['group_id'])
def terminal(m,script,out):
 s=read(m/'started.json');t=read(m/'terminal.json');cmd=s['command'];cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and not t['remaining_process_group'] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()] and cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
def close(x,y):
 if isinstance(x,(list,tuple)):return len(x)==len(y) and all(close(a,b) for a,b in zip(x,y))
 return math.isclose(float(x),float(y),rel_tol=1e-11,abs_tol=1e-10)
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert a.out.resolve()==R/'pixel_floor_counterfactual_review.json';resultdir=D/'result001';terminal(D/'monitor_analysis001',D/'analyze.py',resultdir)
plan=read(D/'plan001.json');out=read(resultdir/'analysis.json');assert out['status']=='completed' and out['GT_opened'] is False and out['new_predictions_opened'] is False and out['plan_sha256']==sha(D/'plan001.json') and out['routes_sha256']==sha(resultdir/'routes.jsonl');assert [(x['name'],x['use_t0_when_t045_k_le']) for x in plan['routes']]==[('fallback_le128',128),('fallback_le256',256),('fallback_le512',512)] and plan['bootstrap']=={'unit':'image_group','paired':True,'replicates':10000,'seed':20261104,'interval':'unadjusted 95% percentile'}
for name,monitor,reviewer in [('t045','monitor_pixel_t045_dev384_scores002','review_pixel_t045_dev384_scores.py'),('t0','monitor_pixel_dev384_scores','review_pixel_dev384_scores.py')]:
 path=B/plan[name+'_review'];score=B/plan[name+'_scores'];review=read(path);terminal(R/monitor,R/reviewer,path);assert review['status']=='passed' and review['independent_review'] is True and review['questions']==384 and review['reviewer_sha256']==sha(R/reviewer) and review['score_sha256']==sha(score) and out['sources'][name+'_scores']==sha(score) and out['sources'][name+'_review']==sha(path)
assert out['sources']['paired_source']==sha(B/'e6/pixelprune_t045_dev384_analysis/paired.py')
x45=rows(B/plan['t045_scores']);x0=rows(B/plan['t0_scores']);index45={key(x):x for x in x45};index0={key(x):x for x in x0};assert len(x45)==len(x0)==len(index45)==len(index0)==384 and set(index45)==set(index0);details=rows(resultdir/'routes.jsonl');indexed={(key(x),x['route']):x for x in details};assert len(details)==len(indexed)==1152
arms=('routed','pixel_t045','pixel_t0','eadp_k256_d1');labels=('routed_minus_t045','routed_minus_t0','routed_minus_eadp_k256');fallback_counts={}
for task in plan['tasks']:
 keys=sorted((k for k in index45 if k[0]==task),key=lambda k:k[1].encode());assert len(keys)==128 and len({k[2] for k in keys})==128;fallback_counts[task]={}
 for route in plan['routes']:
  name=route['name'];threshold=route['use_t0_when_t045_k_le'];values=[];groups=[];answers=[];ks=[];nf=0
  for k in keys:
   q=index45[k];old=index0[k];entry=indexed[(k,name)];k45=q['selected_count'];k0=old['selected_count'];assert type(k45) is type(k0) is int and 1<=k45<=1024 and 1<=k0<=1024
   a45=q['arms']['pixel_t045'];a0=q['arms']['pixel_t0'];o0=old['arms']['pixel_dynamic'];ref=q['arms']['e4_k256_d1'];assert a0['answer']==o0['answer'] and a0['eos']==o0['eos'] and abs(a0['score']-o0['score'])<=1e-12
   fallback=k45<=threshold;chosen=a0 if fallback else a45;chosenk=k0 if fallback else k45;mapping={'routed':chosen,'pixel_t045':a45,'pixel_t0':a0,'eadp_k256_d1':ref};scores={z:mapping[z]['score'] for z in arms};assert all(math.isfinite(v) and 0<=v<=1 for v in scores.values())
   assert entry=={'task':k[0],'qid':k[1],'group_id':k[2],'route':name,'k045':k45,'k0':k0,'fallback_to_t0':fallback,'selected_k':chosenk,'scores':scores};values.append([scores[z] for z in arms]);answers.append([mapping[z].get('answer',mapping[z].get('raw_answer')) for z in arms]);groups.append(k[2]);ks.append(chosenk);nf+=fallback
  rep=out['tasks'][task][name];v=np.array(values,dtype=float);assert rep['n_questions']==rep['n_image_groups']==128 and rep['bootstrap']=={'unit':'image_group','replicates':10000,'seed':20261104,'method':'percentile'}
  order=np.argsort(np.asarray(groups));bygroup=v[order];draw=np.random.default_rng(20261104).integers(0,128,size=(10000,128));boot=bygroup[draw].mean(axis=1)
  for i,arm in enumerate(arms):assert close(rep['arm_means_pct'][arm],100*v[:,i].mean()) and close(rep['arm_percentile_95_ci_pct'][arm],np.quantile(100*boot[:,i],[.025,.975]))
  for j,label in enumerate(labels,1):
   d=v[:,0]-v[:,j];z=rep['contrasts'][label];assert close(z['difference_pp'],100*d.mean()) and close(z['image_cluster_percentile_95_ci_pp'],np.quantile(100*(boot[:,0]-boot[:,j]),[.025,.975])) and z['improved']==int((d>1e-12).sum()) and z['declined']==int((d< -1e-12).sum()) and z['unchanged']==int((abs(d)<=1e-12).sum()) and z['answer_changed']==sum(a[0]!=a[j] for a in answers)
  sk=sorted(ks);assert rep['dynamic_k']=={'min':sk[0],'mean':sum(sk)/128,'median':(sk[63]+sk[64])/2,'p95_nearest_rank':sk[math.ceil(.95*128)-1],'max':sk[-1],'fallbacks':nf};fallback_counts[task][name]=nf
assert out['limitations']==plan['limitations'] and all(any(term in ' '.join(plan['limitations']).lower() for term in terms) for terms in [('identity','reproduce'),('both selectors',),('latency',),('exploratory',)])
proof={'status':'passed','independent_review':True,'questions':384,'route_records':1152,'tasks':3,'routes':3,'contrasts':27,'bootstrap_replicates':10000,'fallback_counts':fallback_counts,'analysis_sha256':sha(resultdir/'analysis.json'),'routes_sha256':sha(resultdir/'routes.jsonl'),'plan_sha256':sha(D/'plan001.json'),'analysis_started_sha256':sha(D/'monitor_analysis001/started.json'),'analysis_terminal_sha256':sha(D/'monitor_analysis001/terminal.json'),'reviewer_sha256':sha(__file__),'GT_opened':False,'new_predictions_opened':False,'limitations':plan['limitations'],'errors':[]}
with a.out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print('passed')
