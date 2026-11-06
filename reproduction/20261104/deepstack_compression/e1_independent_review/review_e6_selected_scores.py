"""Independently recompute complete Div/CD DEV quality scores and paired intervals."""
import argparse,collections,hashlib,io,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'e6/selected_analysis';RUN=ROOT/'e6/selected_full/panel_001';OUT=A/'score002';sys.path.insert(0,str(ROOT/'e1_analysis'));import score,metrics
ARMS=('div:author_stock','div:adapted','cd:author_stock','cd:adapted')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def supervised(start,end,script,out):
 s,t=read(start),read(end);cmd=s['command'];cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid'] and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()] and cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert a.out.resolve()==ROOT/'e1_independent_review/e6_selected_scores_review.json'
 st=A/'monitor_score002/started.json';et=st.parent/'terminal.json';supervised(st,et,A/'score_selected.py',OUT)
 formal=ROOT/'e6/selected_full/panel_001_verification.json';rev=ROOT/'e1_independent_review/selected_full_panel001_review.json';fs=formal.parent/'monitor_verify001/started.json';ft=fs.parent/'terminal.json';rs=ROOT/'e1_independent_review/monitor_selected_full_panel001/started.json';rt=rs.parent/'terminal.json';supervised(fs,ft,formal.parent/'verify.py',formal);supervised(rs,rt,ROOT/'e1_independent_review/review_selected_full_panel.py',rev)
 f=read(formal);r=read(rev);assert f['status']=='passed' and f['questions']==384 and f['generations']==1536 and f['GT_opened'] is False and f['scoring'] is False and r['status']=='passed' and r['independent_review'] is True and r['formal_proof_sha256']==sha(formal) and r['formal_started_sha256']==sha(fs) and r['formal_terminal_sha256']==sha(ft) and r['reviewer_sha256']==sha(ROOT/'e1_independent_review/review_selected_full_panel.py')
 for path,digest in f['files_sha256'].items():assert sha(path)==digest
 report=read(OUT/'analysis.json');frozen=read(OUT/'frozen.json');plan=read(A/'plan.json');assert report['status']=='completed' and report['n_questions']==384 and report['n_predictions']==1536 and report['scores_sha256']==sha(OUT/'scores.jsonl') and frozen['formal_proof_sha256']==sha(formal) and frozen['review_proof_sha256']==sha(rev) and frozen['scorer_prelaunch']['prelaunch_review_sha256']==sha(ROOT/'e1_independent_review/e6_selected_score_prelaunch_review.json') and all(sha(p)==d for p,d in frozen['scorer_prelaunch']['source_sha256'].items())
 ep=read(ROOT/'e1/e1full001_completion.json');dataset=Path(ep['files']['dataset_summary']['path']);assert sha(dataset)==ep['files']['dataset_summary']['sha256'];source=read(dataset)['sources'];records=lines(RUN/'records.jsonl');scores=lines(OUT/'scores.jsonl');assert len(records)==1536 and len(scores)==384 and collections.Counter(x['task'] for x in scores)=={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128}
 pred={(x['task'],str(x['qid']),x['method']+':'+x['path']):x for x in records};assert len(pred)==1536;needed={(x['task'],str(x['qid'])) for x in scores};assert len(needed)==384;truth=score.read_truths(source,needed);author=metrics.author_functions();constants=metrics._author_constants();native={}
 for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
  for arm in ARMS:
   sub=[r for r in records if r['task']==task and r['method']+':'+r['path']==arm];assert len(sub)==128
   stream=io.BytesIO();pd.DataFrame({'index':list(range(len(sub))),'prediction':[r['answer'] for r in sub]}).to_excel(stream,index=False,engine='xlsxwriter');stream.seek(0);back=pd.read_excel(stream);assert list(back['index'])==list(range(128))
   for row,val in zip(sub,back['prediction']):native[(task,str(row['qid']),arm)]=str(val)
 count=0
 for row in scores:
  key=(row['task'],str(row['qid']));assert set(row['arms'])==set(ARMS)
  for arm in ARMS:
   p=pred[(*key,arm)];assert p['group_id']==row['group_id'];v=metrics.score(key[0],p['answer'],truth[key],author,constants,scoring_prediction=native[(*key,arm)]);item=row['arms'][arm];assert item['answer']==p['answer'] and item['eos']==p['eos'] and item['truncated']==p['truncated'] and abs(v['official']-v['independent'])<=1e-12 and abs(item['score']-v['official'])<=1e-12 and abs(item['independent_score']-v['independent'])<=1e-12;count+=1
 contrasts=0
 for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
  group=[r for r in scores if r['task']==task];by=collections.defaultdict(list)
  for row in group:by[row['group_id']].append(row)
  ids=sorted(by);values=np.asarray([[r['arms'][a]['score'] for a in ARMS] for r in group]);sums=np.asarray([[sum(r['arms'][a]['score'] for r in by[g]) for a in ARMS] for g in ids]);sizes=np.asarray([len(by[g]) for g in ids]);draw=np.random.default_rng(plan['bootstrap']['seed']).integers(0,len(ids),(plan['bootstrap']['replicates'],len(ids)));boot=sums[draw].sum(axis=1)/sizes[draw].sum(axis=1)[:,None];out=report['tasks'][task]
  assert out['n_questions']==len(group) and out['n_image_groups']==len(ids)
  for i,arm in enumerate(ARMS):assert abs(out['arm_means_pct'][arm]-100*values[:,i].mean())<1e-10 and np.allclose(out['arm_percentile_95_ci_pct'][arm],np.quantile(100*boot[:,i],[.025,.975]),rtol=0,atol=1e-10)
  for name,(x,y) in plan['contrasts'].items():
   i,j=ARMS.index(x),ARMS.index(y);d=values[:,i]-values[:,j];ci=np.quantile(100*(boot[:,i]-boot[:,j]),[.025,.975]);v=out['contrasts'][name]
   assert abs(v['difference_pp']-100*d.mean())<1e-10 and np.allclose(v['image_cluster_percentile_95_ci_pp'],ci,rtol=0,atol=1e-10) and v['improved']==int((d>1e-12).sum()) and v['declined']==int((d< -1e-12).sum()) and v['unchanged']==int((abs(d)<=1e-12).sum());contrasts+=1
 for desc in source.values():assert sha(desc['path'])==desc['sha256']
 result={'status':'passed','independent_review':True,'questions':384,'scores_verified':count,'bootstrap_contrasts_verified':contrasts,'GT_scope':'DEV only','CONFIRM_FINAL_answers_parsed':False,'scores_sha256':sha(OUT/'scores.jsonl'),'analysis_sha256':sha(OUT/'analysis.json'),'score_started_sha256':sha(st),'score_terminal_sha256':sha(et),'formal_sha256':sha(formal),'panel_review_sha256':sha(rev),'reviewer_sha256':sha(__file__)}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','scores_verified':count,'bootstrap_contrasts_verified':contrasts}))
if __name__=='__main__':main()
