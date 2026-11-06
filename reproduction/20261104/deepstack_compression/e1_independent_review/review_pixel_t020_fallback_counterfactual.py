"""Independently rebuild all posthoc DEV fallback routes and paired intervals."""
import hashlib,json,os
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parent;B=R.parent;D=B/'e6/pixel_t020_fallback_counterfactual'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(1048576),b''):h.update(v)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def key(x):return (x['task'],str(x['qid']),x['group_id'])
def checked_monitor(p):
 s=read(p/'started.json');t=read(p/'terminal.json')
 if t['reason']!='completed' or t['worker_returncode']!=0 or t['remaining_process_group']!=[] or s['worker_pid']!=t['worker_pid']:raise ValueError('incomplete monitor')
def close(x,y):
 if abs(float(x)-float(y))>1e-10:raise ValueError('numeric mismatch')
def main():
 plan=read(D/'plan001.json');frozen=read(D/'source_freeze001.json');result=read(D/'result001/result.json')
 for p in (D/'monitor_cpu001',D/'monitor_freeze001',D/'monitor_run001'):checked_monitor(p)
 if any(sha(p)!=h for p,h in frozen['source_sha256'].items()):raise ValueError('frozen evidence changed')
 if plan['thresholds']!=[128,256,512] or result['questions']!=384 or result['routes_sha256']!=sha(D/'result001/routes.jsonl'):raise ValueError('plan/output mismatch')
 newp=B/'e6/pixelprune_t020_dev384_analysis/score001/scores.jsonl';oldp=B/'e6/pixelprune_dev384_analysis/score001/scores.jsonl'
 newrows=lines(newp);oldrows=lines(oldp);routes=lines(D/'result001/routes.jsonl')
 new={key(x):x for x in newrows};old={key(x):x for x in oldrows};record={key(x):x for x in routes}
 if any(len(x)!=384 for x in (newrows,oldrows,routes,new,old,record)) or set(new)!=set(old) or set(new)!=set(record):raise ValueError('paired inventory mismatch')
 nroute=0
 for k in new:
  a=new[k];b=old[k];r=record[k]
  vals={'k020':a['selected_count'],'k0':b['selected_count'],'s020':a['arms']['pixel_t020']['score'],'s0':b['arms']['pixel_dynamic']['score'],'e4':a['arms']['e4_k256_d1']['score']}
  for field,v in vals.items():close(r[field],v)
  for limit in plan['thresholds']:
   chosen=b if vals['k020']<=limit else a;expected={'source':'t0' if chosen is b else 't020','selected_k':vals['k0'] if chosen is b else vals['k020'],'score':vals['s0'] if chosen is b else vals['s020']};actual=r['routes'][str(limit)]
   if actual['source']!=expected['source']:raise ValueError('route source mismatch')
   close(actual['selected_k'],expected['selected_k']);close(actual['score'],expected['score']);nroute+=1
 comparisons=0
 for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
  keys=sorted((k for k in new if k[0]==task),key=lambda k:k[2])
  if len(keys)!=128 or len({k[2] for k in keys})!=128:raise ValueError('task denominator')
  k20=np.array([new[k]['selected_count'] for k in keys]);k0=np.array([old[k]['selected_count'] for k in keys]);s20=np.array([new[k]['arms']['pixel_t020']['score'] for k in keys],dtype=float);s0=np.array([old[k]['arms']['pixel_dynamic']['score'] for k in keys],dtype=float);e4=np.array([new[k]['arms']['e4_k256_d1']['score'] for k in keys],dtype=float)
  rng=np.random.default_rng(20261104);draw=rng.integers(0,128,(10000,128))
  for limit in plan['thresholds']:
   m=k20<=limit;chosen=np.where(m,s0,s20);selected=np.where(m,k0,k20);out=result['tasks'][task][str(limit)]
   if out['n_questions']!=128 or out['n_image_groups']!=128 or out['fallback_count']!=int(m.sum()):raise ValueError('route counts')
   for name,v in [('fallback_percent',100*m.mean()),('mean_selected_k',selected.mean()),('mean_k020',k20.mean()),('mean_k0',k0.mean()),('mean_score_pct',100*chosen.mean())]:close(out[name],v)
   for name,ref in [('versus_t0',s0),('versus_eadp_k256',e4)]:
    d=chosen-ref;q=out['contrasts'][name];interval=np.quantile(100*d[draw].mean(axis=1),[.025,.975]);close(q['difference_pp'],100*d.mean());close(q['image_group_percentile_95_ci_pp'][0],interval[0]);close(q['image_group_percentile_95_ci_pp'][1],interval[1])
    if (q['improved'],q['declined'],q['unchanged'])!=(int((d>1e-12).sum()),int((d< -1e-12).sum()),int((abs(d)<=1e-12).sum())):raise ValueError('change count')
    comparisons+=1
 proof={'status':'passed','independent_review':True,'questions':384,'routes_checked':nroute,'intervals_checked':comparisons,'GT_opened':False,'score_files_containing_predictions_opened':True,'generation_performed':False,'plan_sha256':sha(D/'plan001.json'),'source_freeze_sha256':sha(D/'source_freeze001.json'),'result_sha256':sha(D/'result001/result.json'),'routes_sha256':sha(D/'result001/routes.jsonl'),'reviewer_sha256':sha(__file__),'errors':[],'limitations':plan['limitations']}
 out=R/'pixel_t020_fallback_counterfactual_review.json'
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
