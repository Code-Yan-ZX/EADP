"""Independent item-level and group-bootstrap reproduction, DEV only."""
import argparse,collections,io,json,math,os,sys
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent.parent;R=B/'e1_independent_review';D=B/'e6/pixelprune_dev384_analysis';OUT=D/'score001';sys.path[:0]=[str(B/'e1_analysis'),str(D)]
import admission,completion,metrics,score as e1score
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert a.out.resolve()==R/'pixel_dev384_scores_review.json'
def read(p):return json.loads(p.read_text())
def lines(p):
 with p.open() as f:return [json.loads(x) for x in f if x.strip()]
def key(x):return (x['task'],str(x['qid']),x['group_id'])
def terminal(mon,script):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in s['command'] if Path(x).name==script.name]==[script.resolve()];return s
term=terminal(D/'monitor_score001',D/'score.py');assert (Path(term['cwd'])/term['command'][term['command'].index('--out')+1]).resolve()==OUT
ad=admission.admission();frozen=read(OUT/'frozen.json');report=read(OUT/'analysis.json');assert frozen['scorer_prelaunch']==ad and frozen['plan_sha256']==report['plan_sha256']==admission.sha(admission.PLAN) and report['status']=='completed' and report['questions']==report['pixel_generations']==384 and report['scores_sha256']==admission.sha(OUT/'scores.jsonl')
rows=lines(OUT/'scores.jsonl');assert len(rows)==384 and len({key(x) for x in rows})==384 and collections.Counter(x['task'] for x in rows)=={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128}
plan=read(B/'e6/pixelprune_dev384/plan001.json');anchors={key(x):x for x in plan['anchors']};assert set(anchors)=={key(x) for x in rows};needed={(x['task'],str(x['qid'])) for x in rows};assert len(needed)==384
truth=e1score.read_truths(frozen['references'],needed);author=metrics.author_functions();constants=metrics._author_constants();import pandas as pd
native={}
for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
 sub=[x for x in rows if x['task']==task];raw=[x['arms']['pixel_dynamic']['answer'] for x in sub];f=io.BytesIO();pd.DataFrame({'index':list(range(128)),'prediction':raw}).to_excel(f,index=False,engine='xlsxwriter');f.seek(0);frame=pd.read_excel(f);assert len(frame)==128 and list(frame['index'])==list(range(128))
 for x,v in zip(sub,frame['prediction']):native[key(x)]=str(v)
prior_e1={key(x):x for x in lines(B/'e1/e1full001_score/scores.jsonl')};prior_e4={key(x):x for x in lines(B/'e4_analysis/score_panel003/scores.jsonl')};assert set(anchors)==set(prior_e1)==set(prior_e4)
for row in rows:
 k=key(row);obs=read(B/'e6/pixelprune_dev384/run_smoke001/observations'/(row['task']+'_'+str(row['qid'])+'.json'));pix=row['arms']['pixel_dynamic'];assert pix['answer']==obs['decoded'] and row['selected_count']==obs['selected_count'] and pix['eos']==obs['ends_eos'] and pix['truncated'] is (not obs['ends_eos'])
 v=metrics.score(row['task'],pix['answer'],truth[(row['task'],str(row['qid']))],author,constants,scoring_prediction=native[k]);assert all(math.isfinite(v[x]) for x in ('official','independent')) and abs(v['official']-v['independent'])<=1e-12 and abs(v['official']-pix['score'])<=1e-12 and abs(v['independent']-pix['independent_score'])<=1e-12
 for arm,old in (('e1_p3_d1',prior_e1[k]['arms']['P3-D1']),('e4_k256_d1',prior_e4[k]['arms']['K256-D1']),('e4_k1024_d1',prior_e4[k]['arms']['K1024-D1'])):
  assert row['arms'][arm]==old and math.isfinite(old['score']) and math.isfinite(old['independent_score']) and abs(old['score']-old['independent_score'])<=1e-12
names=['pixel_dynamic','e1_p3_d1','e4_k256_d1','e4_k1024_d1'];contrasts={'pixel_minus_e1':1,'pixel_minus_e4_k256':2,'pixel_minus_e4_k1024':3}
def close(a,b):
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
 return math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-10)
for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
 sub=[r for r in rows if r['task']==task];by=collections.defaultdict(list)
 for r in sub:by[r['group_id']].append(r)
 groups=sorted(by);tot=np.array([[sum(z['arms'][n]['score'] for z in by[g]) for n in names] for g in groups]);sizes=np.array([len(by[g]) for g in groups]);rng=np.random.default_rng(20261104);ix=rng.integers(0,len(groups),size=(10000,len(groups)));boot=tot[ix].sum(axis=1)/sizes[ix].sum(axis=1)[:,None];v=np.array([[r['arms'][n]['score'] for n in names] for r in sub]);result=report['tasks'][task]
 assert result['n_questions']==128 and result['n_image_groups']==len(groups)
 for i,n in enumerate(names):assert close(result['arm_means_pct'][n],100*v[:,i].mean()) and close(result['arm_percentile_95_ci_pct'][n],np.quantile(100*boot[:,i],[.025,.975]))
 for label,i in contrasts.items():
  delta=v[:,0]-v[:,i];c=result['contrasts'][label];assert close(c['difference_pp'],100*delta.mean()) and close(c['image_cluster_percentile_95_ci_pp'],np.quantile(100*(boot[:,0]-boot[:,i]),[.025,.975])) and c['improved']==int((delta>1e-12).sum()) and c['declined']==int((delta< -1e-12).sum()) and c['unchanged']==int((abs(delta)<=1e-12).sum())
  def ans(r,n):return r['arms'][n].get('answer',r['arms'][n].get('raw_answer'))
  assert c['answer_changed']==sum(ans(r,names[0])!=ans(r,names[i]) for r in sub)
 k=sorted(r['selected_count'] for r in sub);assert result['dynamic_k']=={'min':k[0],'median':(k[63]+k[64])/2,'max':k[-1]}
 assert result['terminal']=={'eos':sum(r['arms']['pixel_dynamic']['eos'] for r in sub),'truncated':sum(r['arms']['pixel_dynamic']['truncated'] for r in sub),'empty':sum(not r['arms']['pixel_dynamic']['answer'].strip() for r in sub)}
r={'status':'passed','independent_review':True,'questions':384,'pixel_scores_recomputed':384,'comparison_scores_checked':1152,'task_contrasts_recomputed':9,'replicates':10000,'DEV_answers_opened':True,'CONFIRM_FINAL_answers_parsed':False,'score_sha256':admission.sha(OUT/'scores.jsonl'),'analysis_sha256':admission.sha(OUT/'analysis.json'),'frozen_sha256':admission.sha(OUT/'frozen.json'),'score_started_sha256':admission.sha(D/'monitor_score001/started.json'),'score_terminal_sha256':admission.sha(D/'monitor_score001/terminal.json'),'reviewer_sha256':admission.sha(__file__),'limitations':['scoring reuses frozen author and independent metric implementations','post-hoc DEV; unadjusted comparisons; dynamic K not equal compute'], 'errors':[]}
with a.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print('passed')
