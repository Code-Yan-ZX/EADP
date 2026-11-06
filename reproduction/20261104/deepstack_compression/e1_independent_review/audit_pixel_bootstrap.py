"""Recompute Pixel DEV means, paired percentile intervals and K summaries from per-question scores."""
import argparse,collections,hashlib,json,os,statistics
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent.parent;S=B/'e6/pixelprune_dev384_analysis/score001';REVIEW=B/'e1_independent_review/pixel_dev384_scores_review.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
 review=json.loads(REVIEW.read_text());assert review['status']=='passed' and review['DEV_answers_opened'] is True and review['CONFIRM_FINAL_answers_parsed'] is False
 analysis=json.loads((S/'analysis.json').read_text());rows=[json.loads(x) for x in (S/'scores.jsonl').read_text().splitlines()];assert len(rows)==384 and sha(S/'scores.jsonl')==analysis['scores_sha256']
 arms=('pixel_dynamic','e1_p3_d1','e4_k256_d1','e4_k1024_d1');pairs={'pixel_minus_e1':('pixel_dynamic','e1_p3_d1'),'pixel_minus_e4_k256':('pixel_dynamic','e4_k256_d1'),'pixel_minus_e4_k1024':('pixel_dynamic','e4_k1024_d1')};results={}
 for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
  sub=[r for r in rows if r['task']==task];assert len(sub)==128 and len({r['group_id'] for r in sub})==128
  sub.sort(key=lambda r:r['group_id']);v=np.array([[r['arms'][k]['score'] for k in arms] for r in sub]);assert np.isfinite(v).all() and (v>=0).all() and (v<=1).all()
  for row in sub:
   for k in arms:assert abs(row['arms'][k]['score']-row['arms'][k]['independent_score'])<=1e-12
  rng=np.random.default_rng(20261104);indices=rng.integers(0,128,(10000,128));boots=v[indices].mean(axis=1);stored=analysis['tasks'][task]
  for i,k in enumerate(arms):assert abs(float(v[:,i].mean()*100)-stored['arm_means_pct'][k])<1e-9
  effects={}
  for label,(x,y) in pairs.items():
   i,j=arms.index(x),arms.index(y);point=float((v[:,i]-v[:,j]).mean()*100);interval=np.quantile((boots[:,i]-boots[:,j])*100,[.025,.975]).tolist();c=stored['contrasts'][label]
   assert abs(point-c['difference_pp'])<1e-9 and np.allclose(interval,c['image_cluster_percentile_95_ci_pp'],rtol=0,atol=1e-9);effects[label]={'difference_pp':point,'95_percent_ci_pp':interval}
  ks=[r['selected_count'] for r in sub];assert {'min':min(ks),'median':statistics.median(ks),'max':max(ks)}==stored['dynamic_k']
  results[task]={'means':stored['arm_means_pct'],'contrasts':effects,'dynamic_k':stored['dynamic_k']}
 result={'status':'passed','GT_opened':False,'questions':384,'bootstrap_replicates':10000,'seed':20261104,'per_task':results,'scores_sha256':sha(S/'scores.jsonl'),'analysis_sha256':sha(S/'analysis.json'),'previous_review_sha256':sha(REVIEW),'reviewer_sha256':sha(__file__),'limitations':['reuses archived per-question scores; does not independently reopen labels or recompute task metrics','post-hoc DEV comparisons are uncorrected and dynamic K is not equal compute']}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('passed')
if __name__=='__main__':main()
