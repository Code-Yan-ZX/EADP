"""Post-hoc DEV associations; existing mixed scored records parsed for K, no new labels or model."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.stats import rankdata,spearmanr
B=Path(__file__).resolve().parent.parent.parent;D=B/'e6/pixel_k_feature_diagnostic'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def key(r):return(r['task'],str(r['qid']),r['group_id'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();plan=read(D/'plan001.json');fp=B/plan['features'];sp=B/plan['pixel_scores'];fr=read(B/plan['feature_review']);sr=read(B/plan['pixel_review'])
 if fr['status']!='passed' or sr['status']!='passed' or fr['features_sha256']!=sha(fp) or sr['score_sha256']!=sha(sp):raise ValueError('unverified source')
 f=lines(fp);s=lines(sp);idx={key(x):x for x in s}
 if len(f)!=384 or len(idx)!=384 or len(s)!=384 or {key(x) for x in f}!=set(idx):raise ValueError('inventory')
 result={}
 for task in plan['tasks']:
  rows=[x for x in f if x['task']==task]
  if len(rows)!=128:raise ValueError('task inventory')
  k=np.array([idx[key(x)]['selected_count'] for x in rows],dtype=np.int64);positive=k<=64;n=int(positive.sum());stats={}
  for name in plan['features_list']:
   values=np.array([x[name] for x in rows],dtype=np.float64)
   if not np.isfinite(values).all():raise ValueError('feature nonfinite')
   rho=float(spearmanr(values,k).statistic) if len(set(values))>1 and len(set(k))>1 else None
   if n>=2 and 128-n>=2:
    ranks=rankdata(values,method='average');auc=float((ranks[positive].sum()-n*(n+1)/2)/(n*(128-n)))
   else:auc=None
   stats[name]={'spearman_with_k':rho,'higher_feature_auc_k_le64':auc}
  result[task]={'questions':128,'k_le64':n,'statistics':stats}
 with a.out.open('x') as f:json.dump({'status':'completed','plan_sha256':sha(D/'plan001.json'),'features_sha256':sha(fp),'pixel_scores_sha256':sha(sp),'tasks':result,'GT_opened':False,'new_predictions_opened':False,'existing_mixed_score_prediction_records_parsed_for_identity_and_k':True,'limitations':plan['limitations']},f,indent=2);f.write('\n')
 print('completed')
if __name__=='__main__':main()
