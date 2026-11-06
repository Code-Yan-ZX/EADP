"""Re-score accepted second-batch DEV oracle and independently recompute paired CIs."""
import argparse,collections,hashlib,io,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'m3_oracle_batch2_analysis';RUN=ROOT/'m3_oracle_batch2/runtime_001';OUT=A/'score_001';V2=ROOT/'m3_oracle_batch2_verify';REV=ROOT/'e1_independent_review'
sys.path[:0]=[str(A),str(ROOT/'e1_analysis')]
import score_gate,analysis_gate,score,metrics
ARMS=('EADP','MAIN-r8','MULTI-r8','RANDOM-r8','MAIN-r16','MULTI-r16','RANDOM-r16')
CONTRASTS={'MULTI-MAIN-r8':('MULTI-r8','MAIN-r8'),'MULTI-RANDOM-r8':('MULTI-r8','RANDOM-r8'),'MULTI-MAIN-r16':('MULTI-r16','MAIN-r16'),'MULTI-RANDOM-r16':('MULTI-r16','RANDOM-r16')}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def generated_only(path):
 result=[]
 with Path(path).open('rb') as f:
  for raw in f:
   if raw.startswith(b'{"event":"generation_completed",'):result.append(json.loads(raw))
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert a.out.resolve()==REV/'m3batch2_scores_review.json'
 started=A/'monitor_score001/started.json';terminal=A/'monitor_score001/terminal.json';s,t=read(started),read(terminal);cmd=s['command'];cwd=Path(s['cwd'])
 assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid']
 assert [(cwd/x).resolve() for x in cmd if Path(x).name=='score_oracle.py']==[(A/'score_oracle.py').resolve()] and cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==OUT
 pre=analysis_gate.validate();report=read(OUT/'analysis.json');frozen=read(OUT/'frozen.json')
 assert frozen['analysis_approval']==pre and report['status']=='completed' and report['n_questions']==128 and report['n_new_generations']==768 and report['scores_sha256']==sha(OUT/'scores.jsonl') and all(sha(p)==v for p,v in frozen['source_sha256'].items())
 from types import SimpleNamespace
 args=SimpleNamespace(run=RUN,proof=V2/'runtime_001_verification.json',verify_started=V2/'monitor_verify001/started.json',verify_terminal=V2/'monitor_verify001/terminal.json',run_started=ROOT/'m3_oracle_batch2/monitor_runtime001/started.json',run_terminal=ROOT/'m3_oracle_batch2/monitor_runtime001/terminal.json',review=REV/'m3oracle_batch2_panel_review.json',review_started=REV/'monitor_m3batch2_panel001/started.json',review_terminal=REV/'monitor_m3batch2_panel001/terminal.json')
 formal=score_gate.gate(args);assert frozen['acceptance_sha256']==sha(args.proof) and frozen['review_sha256']==sha(args.review)
 # Only after both completed acceptance chains and the score supervisor exit do
 # we parse the accepted predictions or requested DEV reference rows.
 saved=lines(OUT/'scores.jsonl');sample=read(ROOT/'m3_oracle_batch2/sample_002.json')['examples'];assert len(saved)==len(sample)==128 and collections.Counter(x['task'] for x in saved)=={'DocVQA_VAL':64,'TextVQA_VAL':32,'AI2D_TEST':32}
 e1=read(ROOT/'e1/e1full001_completion.json');journal=Path(e1['files']['journal']['path']);assert sha(journal)==e1['files']['journal']['sha256'];baseline={(r['task'],str(r['qid']),r['group_id']):r for r in lines(journal) if r.get('arm')=='P3-D1' and r.get('forced') is False};assert len(baseline)==384
 starts=[x for x in lines(RUN/'panel/panel_events.jsonl') if x['event']=='question_started'];assert len(starts)==128;answers={}
 for row,begin in zip(sample,starts):
  key=(row['task'],str(row['qid']),row['group_id']);assert (begin['task'],str(begin['qid']),begin['group_id'])==key
  path=RUN/'panel'/begin['journal'];assert formal['result_files_sha256'][str(path)]==sha(path);events=generated_only(path);assert len(events)==6
  answers[(*key,'EADP')]=baseline[key]
  for event in events:
   arm=event['source']+'-r'+str(event['r']);k=(*key,arm);assert k not in answers;answers[k]=event['output']
 assert len(answers)==128*7
 native={}
 for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
  rows=[r for r in saved if r['task']==task]
  for arm in ARMS:
   vals=[answers[(task,str(r['qid']),r['group_id'],arm)]['answer'] for r in rows];b=io.BytesIO();pd.DataFrame({'index':list(range(len(vals))),'prediction':vals}).to_excel(b,index=False,engine='xlsxwriter');b.seek(0);back=pd.read_excel(b);assert list(back['index'])==list(range(len(vals)))
   for r,v in zip(rows,back['prediction']):native[(task,str(r['qid']),r['group_id'],arm)]=str(v)
 ds=Path(e1['files']['dataset_summary']['path']);assert sha(ds)==e1['files']['dataset_summary']['sha256'];sources=read(ds)['sources'];keys={(x['task'],str(x['qid'])) for x in sample};assert len(keys)==128
 truth=score.read_truths(sources,keys);author=metrics.author_functions();constants=metrics._author_constants();verified=0
 for row in saved:
  key=(row['task'],str(row['qid']),row['group_id']);assert set(row['arms'])==set(ARMS)
  for arm in ARMS:
   raw=answers[(*key,arm)];item=row['arms'][arm];assert all(raw[k]==item[k] for k in ('answer','eos','truncated'))
   result=metrics.score(key[0],raw['answer'],truth[(key[0],key[1])],author,constants,scoring_prediction=native[(*key,arm)])
   assert abs(result['official']-result['independent'])<=1e-12 and abs(item['score']-result['official'])<=1e-12 and abs(item['independent_score']-result['independent'])<=1e-12;verified+=1
 for label in ('ALL','DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
  rows=saved if label=='ALL' else [r for r in saved if r['task']==label];by=collections.defaultdict(list)
  for r in rows:by[r['group_id']].append(r)
  groups=sorted(by);values=np.asarray([[r['arms'][arm]['score'] for arm in ARMS] for r in rows]);sums=np.asarray([[sum(r['arms'][arm]['score'] for r in by[g]) for arm in ARMS] for g in groups]);sizes=np.asarray([len(by[g]) for g in groups]);draw=np.random.default_rng(20261104).integers(0,len(groups),(10000,len(groups)));boot=sums[draw].sum(axis=1)/sizes[draw].sum(axis=1)[:,None];out=report['reports'][label]
  assert out['n_questions']==len(rows) and out['n_image_groups']==len(groups)
  for i,arm in enumerate(ARMS):
   assert abs(out['means_pct'][arm]-100*values[:,i].mean())<1e-10
   terminal_stats=out['terminal'][arm];assert terminal_stats=={'eos':sum(r['arms'][arm]['eos'] for r in rows),'truncated':sum(r['arms'][arm]['truncated'] for r in rows),'empty':sum(not r['arms'][arm]['answer'].strip() for r in rows)}
  for name,(a1,b1) in CONTRASTS.items():
   i,j=ARMS.index(a1),ARMS.index(b1);diff=values[:,i]-values[:,j];ci=np.quantile(100*(boot[:,i]-boot[:,j]),[.025,.975]);c=out['contrasts'][name]
   assert abs(c['difference_pp']-100*diff.mean())<1e-10 and np.allclose(c['image_group_percentile_95_ci_pp'],ci,rtol=0,atol=1e-10)
   assert c['improved']==int((diff>1e-12).sum()) and c['declined']==int((diff< -1e-12).sum()) and c['unchanged']==int((abs(diff)<=1e-12).sum()) and c['answer_changed']==sum(r['arms'][a1]['answer']!=r['arms'][b1]['answer'] for r in rows) and c['minimum_relevant_gain_pp']==1.0 and c['ci_excludes_minimum_relevant_gain']==bool(ci[1]<1.)
 for desc in sources.values():assert sha(desc['path'])==desc['sha256']
 result={'status':'passed','independent_review':True,'questions':128,'scores_verified':verified,'bootstrap_contrasts_verified':16,'GT_scope':'DEV only','CONFIRM_FINAL_answers_parsed':False,
         'scores_sha256':sha(OUT/'scores.jsonl'),'analysis_sha256':sha(OUT/'analysis.json'),'score_started_sha256':sha(started),'score_terminal_sha256':sha(terminal),'formal_sha256':sha(args.proof),'panel_review_sha256':sha(args.review),'reviewer_sha256':sha(__file__)}
 with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'status':'passed','questions':128,'scores_verified':verified,'bootstrap_contrasts_verified':16}))
if __name__=='__main__':main()
