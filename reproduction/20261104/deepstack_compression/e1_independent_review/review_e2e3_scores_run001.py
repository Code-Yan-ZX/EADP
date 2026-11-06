"""Independent DEV-only rescore and bootstrap check after full E2/E3 completion."""
import argparse,hashlib,io,json,math,sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parent.parent; ANALYSIS=BASE/'e2_e3_analysis';RUN=BASE/'e2_e3_full/run_001';SCORES=ANALYSIS/'score_run001'
sys.path.insert(0,str(BASE/'e1_analysis'))
import score,metrics
ARMS=tuple('E2/'+x for x in ('000','100','010','001','110','101','011','111'))+('E3/PERM','E3/MEAN')
CONTRASTS={'E2_leave_slot0':{'E2/111':1,'E2/011':-1},'E2_leave_slot1':{'E2/111':1,'E2/101':-1},'E2_leave_slot2':{'E2/111':1,'E2/110':-1},
 'E2_only_slot0':{'E2/100':1,'E2/000':-1},'E2_only_slot1':{'E2/010':1,'E2/000':-1},'E2_only_slot2':{'E2/001':1,'E2/000':-1},
 'E2_three_way':{'E2/111':1,'E2/110':-1,'E2/101':-1,'E2/011':-1,'E2/100':1,'E2/010':1,'E2/001':1,'E2/000':-1},
 'E3_match_minus_perm':{'E2/111':1,'E3/PERM':-1},'E3_match_minus_mean':{'E2/111':1,'E3/MEAN':-1},
 'E3_perm_minus_off':{'E3/PERM':1,'E2/000':-1},'E3_mean_minus_off':{'E3/MEAN':1,'E2/000':-1}}
PRIMARY={'E2_leave_slot0','E2_leave_slot1','E2_leave_slot2'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def lines(path):
 data=Path(path).read_bytes();assert not data or data.endswith(b'\n');return [json.loads(x) for x in data.splitlines()]
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 start=read(ANALYSIS/'monitor_score_run001/started.json');end=read(ANALYSIS/'monitor_score_run001/terminal.json');cmd=start['command'];cwd=Path(start['cwd'])
 assert end['reason']=='completed' and end['worker_returncode']==0 and end['remaining_process_group']==[] and start['worker_pid']==end['worker_pid']
 assert [(cwd/x).resolve() for x in cmd if Path(x).name=='score_panel.py']==[(ANALYSIS/'score_panel.py').resolve()] and cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==SCORES
 report=read(SCORES/'analysis.json');frozen=read(SCORES/'frozen.json');assert report['status']=='completed' and report['n_questions']==384 and report['n_predictions']==3840
 assert report['scores_sha256']==sha(SCORES/'scores.jsonl') and all(sha(path)==digest for path,digest in frozen['source_sha256'].items())
 formal=BASE/'e2_e3_full/run_001_verification.json';review=BASE/'e1_independent_review/e2e3full_run001_review.json'
 assert frozen['panel_proof_sha256']==sha(formal) and frozen['review_proof_sha256']==sha(review)
 fp=read(formal);rp=read(review);assert fp['status']=='passed' and rp['status']=='passed' and rp['formal_sha256']==sha(formal) and fp['GT_opened'] is False and rp['GT_opened'] is False
 e1=read(BASE/'e1/e1full001_completion.json');dataset=Path(e1['files']['dataset_summary']['path']);assert sha(dataset)==e1['files']['dataset_summary']['sha256'];source=read(dataset)['sources']
 saved=lines(SCORES/'scores.jsonl');records=lines(RUN/'data/records.jsonl');assert len(saved)==384 and len(records)==3840
 keys={(x['task'],str(x['qid'])) for x in saved};assert len(keys)==384 and Counter(x['task'] for x in saved)=={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128}
 truth=score.read_truths(source,keys);author=metrics.author_functions();constants=metrics._author_constants();bykey={}
 for r in records:
  key=(r['task'],str(r['qid']),r['family']+'/'+r['arm']);assert key not in bykey;bykey[key]=r
 native={}
 for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
  rows=[x for x in saved if x['task']==task]
  for arm in ARMS:
   data=[bykey[(task,str(x['qid']),arm)]['answer'] for x in rows];stream=io.BytesIO()
   pd.DataFrame({'index':list(range(128)),'prediction':data}).to_excel(stream,index=False,engine='xlsxwriter');stream.seek(0);back=pd.read_excel(stream);assert list(back['index'])==list(range(128))
   for row,value in zip(rows,back['prediction']):native[(task,str(row['qid']),arm)]=str(value)
 n=0
 for row in saved:
  key=(row['task'],str(row['qid']));assert set(row['arms'])==set(ARMS)
  for arm in ARMS:
   record=bykey[(*key,arm)];entry=row['arms'][arm];assert entry['answer']==record['answer'] and entry['eos']==record['eos'] and entry['truncated']==record['truncated']
   value=metrics.score(key[0],record['answer'],truth[key],author,constants,scoring_prediction=native[(*key,arm)])
   assert abs(value['official']-value['independent'])<=1e-12 and abs(entry['score']-value['official'])<=1e-12 and abs(entry['independent_score']-value['independent'])<=1e-12;n+=1
 for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
  rows=[x for x in saved if x['task']==task];ids=sorted({x['group_id'] for x in rows});assert len(ids)==128
  group={x['group_id']:x for x in rows};values=np.array([[group[g]['arms'][a]['score'] for a in ARMS] for g in ids]);draw=np.random.default_rng(20261104).integers(0,len(ids),(10000,len(ids)));boot=values[draw].mean(axis=1);summ=report['tasks'][task]
  for i,arm in enumerate(ARMS):assert abs(summ['arm_means_pct'][arm]-100*values[:,i].mean())<1e-10
  for name,weights in CONTRASTS.items():
   v=np.array([weights.get(arm,0) for arm in ARMS]);observed=values@v;sampled=100*(boot@v);alpha=.05/3 if name in PRIMARY else .05;ci=np.quantile(sampled,[alpha/2,1-alpha/2]);q=summ['contrasts'][name]
   assert abs(q['difference_pp']-100*observed.mean())<1e-10 and np.allclose(q['percentile_ci_pp'],ci,rtol=0,atol=1e-10)
   assert q['positive_questions']==int((observed>1e-12).sum()) and q['negative_questions']==int((observed< -1e-12).sum()) and q['zero_questions']==int((abs(observed)<=1e-12).sum())
 for x in source.values():assert sha(x['path'])==x['sha256']
 out={'status':'passed','independent_review':True,'questions':384,'scores_verified':n,'bootstrap_contrasts_verified':33,'GT_scope':'DEV only','CONFIRM_FINAL_answers_parsed':False,
      'analysis_sha256':sha(SCORES/'analysis.json'),'scores_sha256':sha(SCORES/'scores.jsonl'),'score_started_sha256':sha(ANALYSIS/'monitor_score_run001/started.json'),
      'score_terminal_sha256':sha(ANALYSIS/'monitor_score_run001/terminal.json'),'formal_sha256':sha(formal),'panel_review_sha256':sha(review),'reviewer_sha256':sha(__file__)}
 with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','questions':384,'scores_verified':n,'bootstrap_contrasts_verified':33}))
if __name__=='__main__':main()
