"""Independent full DEV PriorTR/E1 score and paired-bootstrap recomputation."""
import argparse,collections,hashlib,io,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent.parent;D=ROOT/'e6/priortr/dev384';A=D/'analysis';RUN=D/'run_001';OUT=A/'score_001';sys.path.insert(0,str(ROOT/'e1_analysis'))
import score,metrics
ARMS=('stock_native','author_default','priortr_k256','EADP_P3_D1_reused_E1')
CONTRASTS=(('author_default','stock_native'),('priortr_k256','stock_native'),('author_default',ARMS[-1]),('priortr_k256',ARMS[-1]))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def supervisor(start,end,script,out=None):
 s,t=read(start),read(end);cwd=Path(s['cwd']);cmd=s['command'];assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid']
 assert [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()]
 if out is not None:assert cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert a.out.resolve()==ROOT/'e1_independent_review/priortr384_scores_review.json'
 st=A/'monitor_score001/started.json';et=st.parent/'terminal.json';supervisor(st,et,A/'score_panel.py',OUT)
 formal=D/'run_001_verification.json';review=D/'run_001_independent_review.json';fs=D/'monitor_verify001/started.json';ft=fs.parent/'terminal.json';rs=D/'monitor_review_run001/started.json';rt=rs.parent/'terminal.json'
 supervisor(fs,ft,D/'panel_verify.py',formal);supervisor(rs,rt,D/'review_panel.py',review)
 f=read(formal);r=read(review);assert f['status']=='passed' and f['questions']==384 and f['generations']==1152 and f['GT_opened'] is False and f['scoring_performed'] is False and r['status']=='passed' and r['independent_review'] is True and r['formal_sha256']==sha(formal) and r['formal_started_sha256']==sha(fs) and r['formal_terminal_sha256']==sha(ft) and r['reviewer_sha256']==sha(D/'review_panel.py')
 for path,digest in {**f['files_sha256'],**f['array_sha256']}.items():assert sha(path)==digest
 rep=read(OUT/'analysis.json');frozen=read(OUT/'frozen.json');assert rep['status']=='completed' and rep['questions']==384 and rep['new_predictions']==1152 and rep['scores_sha256']==sha(OUT/'scores.jsonl') and frozen['prior_formal_sha256']==sha(formal) and frozen['prior_review_sha256']==sha(review)
 assert all(sha(path)==digest for path,digest in frozen['source_sha256'].items()) and frozen['score_source_review_sha256']==sha(A/'prelaunch_review.json')
 e1=read(ROOT/'e1/e1full001_completion.json');journal=Path(e1['files']['journal']['path']);dataset=Path(e1['files']['dataset_summary']['path']);assert sha(journal)==e1['files']['journal']['sha256'] and sha(dataset)==e1['files']['dataset_summary']['sha256']
 # The two clean panel acceptances and score supervisor have now completed;
 # only the selected DEV qids below will have references parsed.
 sample=read(D/'sample_001.json')['sample'];scores=lines(OUT/'scores.jsonl');pred=lines(RUN/'data/records.jsonl');assert len(sample)==len(scores)==384 and len(pred)==1152
 assert collections.Counter(r['task'] for r in scores)=={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128}
 e1rows=[r for r in lines(journal) if r.get('arm')=='P3-D1' and r.get('forced') is False];assert len(e1rows)==384
 by={(r['task'],str(r['qid']),r['arm']):r for r in pred};base={(r['task'],str(r['qid'])):r for r in e1rows};assert len(by)==1152 and len(base)==384
 source=read(dataset)['sources'];needed={(r['task'],str(r['qid'])) for r in sample};assert len(needed)==384;truth=score.read_truths(source,needed);author=metrics.author_functions();constants=metrics._author_constants();native={}
 for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
  group=[r for r in scores if r['task']==task]
  for arm in ARMS:
   vals=[]
   for row in group:
    key=(task,str(row['qid']));record=base[key] if arm==ARMS[-1] else by[(*key,arm)];vals.append(record['answer'])
   stream=io.BytesIO();pd.DataFrame({'index':list(range(len(vals))),'prediction':vals}).to_excel(stream,index=False,engine='xlsxwriter');stream.seek(0);back=pd.read_excel(stream);assert list(back['index'])==list(range(len(vals)))
   for row,value in zip(group,back['prediction']):native[(task,str(row['qid']),arm)]=str(value)
 count=0
 for row in scores:
  key=(row['task'],str(row['qid']));assert set(row['arms'])==set(ARMS)
  for arm in ARMS:
   record=base[key] if arm==ARMS[-1] else by[(*key,arm)];assert record['group_id']==row['group_id']
   got=metrics.score(key[0],record['answer'],truth[key],author,constants,scoring_prediction=native[(*key,arm)]);item=row['arms'][arm]
   assert abs(got['official']-got['independent'])<=1e-12 and abs(item['score']-got['official'])<=1e-12 and abs(item['independent_score']-got['independent'])<=1e-12 and item['eos']==record['eos'] and item['truncated']==record['truncated'];count+=1
 for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
  group=[r for r in scores if r['task']==task];bygroup=collections.defaultdict(list)
  for r in group:bygroup[r['group_id']].append(r)
  ids=sorted(bygroup);values=np.asarray([[r['arms'][arm]['score'] for arm in ARMS] for r in group]);sums=np.asarray([[sum(r['arms'][arm]['score'] for r in bygroup[g]) for arm in ARMS] for g in ids]);sizes=np.asarray([len(bygroup[g]) for g in ids]);draw=np.random.default_rng(20261104).integers(0,len(ids),(10000,len(ids)));boot=sums[draw].sum(axis=1)/sizes[draw].sum(axis=1)[:,None];out=rep['tasks'][task]
  assert out['n_questions']==len(group) and out['n_image_groups']==len(ids)
  for i,arm in enumerate(ARMS):assert abs(out['means_pct'][arm]-100*values[:,i].mean())<1e-10
  for x,y in CONTRASTS:
   i,j=ARMS.index(x),ARMS.index(y);diff=values[:,i]-values[:,j];ci=np.quantile(100*(boot[:,i]-boot[:,j]),[.025,.975]);v=out['contrasts'][x+' - '+y]
   assert abs(v['difference_pp']-100*diff.mean())<1e-10 and np.allclose(v['percentile_95_ci_pp'],ci,rtol=0,atol=1e-10) and v['positive_questions']==int((diff>1e-12).sum()) and v['negative_questions']==int((diff< -1e-12).sum()) and v['zero_questions']==int((abs(diff)<=1e-12).sum())
 for desc in source.values():assert sha(desc['path'])==desc['sha256']
 result={'status':'passed','independent_review':True,'questions':384,'scores_verified':count,'bootstrap_contrasts_verified':12,'GT_scope':'DEV only','CONFIRM_FINAL_answers_parsed':False,
         'scores_sha256':sha(OUT/'scores.jsonl'),'analysis_sha256':sha(OUT/'analysis.json'),'score_started_sha256':sha(st),'score_terminal_sha256':sha(et),'formal_sha256':sha(formal),'panel_review_sha256':sha(review),'reviewer_sha256':sha(__file__)}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','questions':384,'scores_verified':count,'bootstrap_contrasts_verified':12}))
if __name__=='__main__':main()
