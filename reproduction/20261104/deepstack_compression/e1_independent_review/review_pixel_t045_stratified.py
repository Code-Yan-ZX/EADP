"""Independently reconstruct post-hoc K strata from accepted DEV scores."""
import argparse,collections,hashlib,json,math,os
from pathlib import Path
B=Path(__file__).resolve().parent.parent;R=B/'e1_independent_review';D=B/'e6/pixelprune_t045_stratified'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def terminal(m,script,out):
 s=read(m/'started.json');t=read(m/'terminal.json');cmd=s['command'];cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and not t['remaining_process_group'] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()] and cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
def close(a,b):return a is b if a is None or b is None else math.isclose(a,b,rel_tol=1e-13,abs_tol=1e-11)
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();assert a.out.resolve()==R/'pixel_t045_stratified_review.json'
plan=read(D/'plan001.json');out=read(D/'analysis001.json');score=B/plan['score_path'];proof=B/plan['independent_review'];r=read(proof);terminal(D/'monitor_analysis001',D/'analyze.py',D/'analysis001.json');terminal(R/'monitor_pixel_t045_dev384_scores002',R/'review_pixel_t045_dev384_scores.py',proof)
assert out['status']=='completed' and out['GT_opened'] is False and out['new_predictions_opened'] is False and out['plan_sha256']==sha(D/'plan001.json') and out['scores_sha256']==r['score_sha256']==sha(score) and out['independent_review_sha256']==sha(proof) and r['status']=='passed' and r['independent_review'] is True and r['questions']==384 and r['reviewer_sha256']==sha(R/'review_pixel_t045_dev384_scores.py')
bins=plan['bins_inclusive'];assert bins==[[1,64],[65,128],[129,256],[257,512],[513,1024]] and {k for lo,hi in bins for k in range(lo,hi+1)}==set(range(1,1025));rows=[json.loads(x) for x in score.read_text().splitlines() if x.strip()];assert len(rows)==384 and len({(x['task'],str(x['qid']),x['group_id']) for x in rows})==384 and collections.Counter(x['task'] for x in rows)=={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128}
summary={}
for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
 sub=[x for x in rows if x['task']==task];counts=[];assert len(out['tasks'][task])==len(bins)
 for (lo,hi),item in zip(bins,out['tasks'][task]):
  selected=[x for x in sub if type(x['selected_count']) is int and lo<=x['selected_count']<=hi];n=len(selected);assert item['range']==[lo,hi] and item['count']==n
  for x in selected:
   for arm in ('pixel_t045','pixel_t0'):
    z=x['arms'][arm];assert math.isfinite(z['score']) and math.isfinite(z['independent_score']) and 0<=z['score']<=1 and abs(z['score']-z['independent_score'])<=1e-12
   z=x['arms']['pixel_t045'];assert type(z['eos']) is bool and type(z['truncated']) is bool and z['eos']!=z['truncated']
  m45=sum(x['arms']['pixel_t045']['score'] for x in selected)*100/n if n else None;m0=sum(x['arms']['pixel_t0']['score'] for x in selected)*100/n if n else None;delta=sum(x['arms']['pixel_t045']['score']-x['arms']['pixel_t0']['score'] for x in selected)*100/n if n else None
  assert close(item['mean_t045_pct'],m45) and close(item['mean_t0_on_same_items_pct'],m0) and close(item['paired_difference_pp'],delta) and item['eos']==sum(x['arms']['pixel_t045']['eos'] for x in selected) and item['truncated']==sum(x['arms']['pixel_t045']['truncated'] for x in selected);counts.append(n)
 assert sum(counts)==128;summary[task]=counts
outproof={'status':'passed','independent_review':True,'questions':384,'task_bin_counts':summary,'analysis_sha256':sha(D/'analysis001.json'),'analysis_started_sha256':sha(D/'monitor_analysis001/started.json'),'analysis_terminal_sha256':sha(D/'monitor_analysis001/terminal.json'),'plan_sha256':sha(D/'plan001.json'),'score_review_sha256':sha(proof),'reviewer_sha256':sha(__file__),'GT_opened':False,'new_predictions_opened':False,'limitations':['post-hoc bins chosen after observing DEV; descriptive, not causal','empty bins have null means and small bins provide weak evidence','scores and threshold chosen on the same DEV data; no new quality or equal-compute claim'],'errors':[]}
with a.out.open('x') as f:json.dump(outproof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print('passed')
