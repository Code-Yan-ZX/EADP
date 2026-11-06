"""Independent per-question and image-group bootstrap verification, DEV only."""
import argparse,collections,io,json,math,sys
from pathlib import Path
import numpy as np,pandas as pd
B=Path(__file__).resolve().parent.parent;I=B/'e1_independent_review';D=B/'e6/stacktok_full_analysis';OUT=D/'score001';sys.path[:0]=[str(B/'e1_analysis'),str(D)]
import admission,completion,metrics,score as e1score
read=lambda p:json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def key(x):return (x['task'],str(x['qid']),x['group_id'])
def term(m,script,out):
 s=read(m/'started.json');t=read(m/'terminal.json');c=s['command'];cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in c if Path(x).name==script.name]==[script.resolve()] and c.count('--out')==1 and (cwd/c[c.index('--out')+1]).resolve()==out.resolve()
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert a.out.resolve()==I/'stacktok_full_scores_review.json';term(D/'monitor_score001',D/'score.py',OUT)
proof=admission.require();frozen=read(OUT/'frozen.json');analysis=read(OUT/'analysis.json');assert frozen['prelaunch']==proof and frozen['plan_sha256']==analysis['plan_sha256']==admission.sha(D/'score_plan001.json') and analysis['status']=='completed' and analysis['questions']==384 and analysis['scores_sha256']==admission.sha(OUT/'scores.jsonl')
rows=lines(OUT/'scores.jsonl');records={key(x):x for x in lines(B/'e6/stacktok_full/run_001/records.jsonl')};prior={key(x):x for x in lines(B/'e6/pixelprune_t045_dev384_analysis/score001/scores.jsonl')};anchors={key(x):x for x in read(B/'e6/pixelprune_t045_dev384/plan001.json')['anchors']};assert len(rows)==len(records)==len(prior)==len(anchors)==384 and {key(x) for x in rows}==set(records)==set(prior)==set(anchors) and collections.Counter(x['task'] for x in rows)=={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128}
needed={(x['task'],str(x['qid'])) for x in rows};assert len(needed)==384;truth=e1score.read_truths(frozen['references'],needed);author=metrics.author_functions();constants=metrics._author_constants();native={}
for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
 sub=[x for x in rows if x['task']==task];data=io.BytesIO();pd.DataFrame({'index':list(range(128)),'prediction':[x['arms']['stacktok_k256']['answer'] for x in sub]}).to_excel(data,index=False,engine='xlsxwriter');data.seek(0);frame=pd.read_excel(data);assert len(frame)==128 and list(frame['index'])==list(range(128))
 for row,v in zip(sub,frame['prediction']):native[key(row)]=str(v)
for row in rows:
 k=key(row);x=records[k];mine=row['arms']['stacktok_k256'];assert mine['answer']==x['answer'] and mine['eos']==x['eos'] and mine['truncated']==x['truncated'];v=metrics.score(row['task'],mine['answer'],truth[(row['task'],str(row['qid']))],author,constants,scoring_prediction=native[k]);assert all(math.isfinite(v[q]) for q in ('official','independent')) and abs(v['official']-v['independent'])<=1e-12 and abs(v['official']-mine['score'])<=1e-12 and abs(v['independent']-mine['independent_score'])<=1e-12
 for new,old in (('eadp_k256_d1','e4_k256_d1'),('pixel_t0','pixel_t0'),('pixel_t045','pixel_t045')):assert row['arms'][new]==prior[k]['arms'][old] and math.isfinite(row['arms'][new]['score']) and abs(row['arms'][new]['score']-row['arms'][new]['independent_score'])<=1e-12
names=['stacktok_k256','eadp_k256_d1','pixel_t0','pixel_t045'];contrasts={'stacktok_minus_eadp':1,'stacktok_minus_pixel_t0':2,'stacktok_minus_pixel_t045':3}
def close(x,y):
 if isinstance(x,(list,tuple)):return len(x)==len(y) and all(close(u,v) for u,v in zip(x,y))
 return math.isclose(float(x),float(y),rel_tol=1e-11,abs_tol=1e-10)
for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
 sub=[r for r in rows if r['task']==task];grouped=collections.defaultdict(list)
 for row in sub:grouped[row['group_id']].append(row)
 groups=sorted(grouped);sums=np.array([[sum(z['arms'][n]['score'] for z in grouped[g]) for n in names] for g in groups]);sizes=np.array([len(grouped[g]) for g in groups]);rng=np.random.default_rng(20261104);idx=rng.integers(0,len(groups),(10000,len(groups)));boot=sums[idx].sum(axis=1)/sizes[idx].sum(axis=1)[:,None];v=np.array([[r['arms'][n]['score'] for n in names] for r in sub]);result=analysis['tasks'][task];assert result['n_questions']==128 and result['n_image_groups']==len(groups)
 for i,n in enumerate(names):assert close(result['arm_means_pct'][n],100*v[:,i].mean()) and close(result['arm_percentile_95_ci_pct'][n],np.quantile(100*boot[:,i],[.025,.975]))
 for label,i in contrasts.items():
  d=v[:,0]-v[:,i];c=result['contrasts'][label];assert close(c['difference_pp'],100*d.mean()) and close(c['image_cluster_percentile_95_ci_pp'],np.quantile(100*(boot[:,0]-boot[:,i]),[.025,.975])) and c['improved']==int((d>1e-12).sum()) and c['declined']==int((d< -1e-12).sum()) and c['unchanged']==int((abs(d)<=1e-12).sum()) and c['answer_changed']==sum(r['arms'][names[0]]['answer']!=r['arms'][names[i]].get('answer',r['arms'][names[i]].get('raw_answer')) for r in sub)
 assert result['terminal']=={'eos':sum(r['arms']['stacktok_k256']['eos'] for r in sub),'truncated':sum(r['arms']['stacktok_k256']['truncated'] for r in sub),'empty':sum(not r['arms']['stacktok_k256']['answer'].strip() for r in sub)}
out={'status':'passed','independent_review':True,'questions':384,'stacktok_scores_recomputed':384,'reference_scores_checked':1152,'task_contrasts_recomputed':9,'replicates':10000,'DEV_answers_opened':True,'CONFIRM_FINAL_answers_parsed':False,'score_sha256':admission.sha(OUT/'scores.jsonl'),'analysis_sha256':admission.sha(OUT/'analysis.json'),'frozen_sha256':admission.sha(OUT/'frozen.json'),'score_started_sha256':admission.sha(D/'monitor_score001/started.json'),'score_terminal_sha256':admission.sha(D/'monitor_score001/terminal.json'),'reviewer_sha256':admission.sha(__file__),'limitations':['the scoring implementations are the already accepted E1 author and independent metrics','post-hoc exploratory DEV comparison; 95% intervals unadjusted and same K does not establish equal compute'], 'errors':[]};a.out.write_text(json.dumps(out,indent=2)+'\n');print('passed')
