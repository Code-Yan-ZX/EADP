"""Descriptive post-hoc K strata from independently accepted per-item scores."""
import argparse,hashlib,json,math
from pathlib import Path
B=Path(__file__).resolve().parent.parent.parent;D=B/'e6/pixelprune_t045_stratified'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();plan=json.loads((D/'plan001.json').read_text());source=B/plan['score_path'];proof=json.loads((B/plan['independent_review']).read_text())
 if proof['status']!='passed' or proof.get('independent_review') is not True or proof['score_sha256']!=sha(source) or proof['questions']!=384:raise ValueError('score proof failed')
 rows=[json.loads(x) for x in source.read_text().splitlines()]
 if len(rows)!=384 or len({(r['task'],r['qid'],r['group_id']) for r in rows})!=384:raise ValueError('question identity')
 tasks={}
 for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
  sub=[r for r in rows if r['task']==task]
  if len(sub)!=128:raise ValueError('task denominator')
  result=[]
  for low,high in plan['bins_inclusive']:
   selected=[r for r in sub if low<=r['selected_count']<=high];n=len(selected)
   for r in selected:
    for arm in ('pixel_t045','pixel_t0'):
     v=r['arms'][arm]['score'];assert math.isfinite(v) and 0<=v<=1
   def mean(arm):return None if not n else sum(x['arms'][arm]['score'] for x in selected)*100/n
   a1,a0=mean('pixel_t045'),mean('pixel_t0')
   result.append({'range':[low,high],'count':n,'mean_t045_pct':a1,'mean_t0_on_same_items_pct':a0,'paired_difference_pp':None if not n else a1-a0,'eos':sum(r['arms']['pixel_t045']['eos'] for r in selected),'truncated':sum(r['arms']['pixel_t045']['truncated'] for r in selected)})
  if sum(x['count'] for x in result)!=128:raise ValueError('bin loss')
  tasks[task]=result
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x') as f:json.dump({'status':'completed','plan_sha256':sha(D/'plan001.json'),'scores_sha256':sha(source),'independent_review_sha256':sha(B/plan['independent_review']),'tasks':tasks,'GT_opened':False,'new_predictions_opened':False,'limitations':plan['limitations']},f,indent=2);f.write('\n')
 print('completed')
if __name__=='__main__':main()
