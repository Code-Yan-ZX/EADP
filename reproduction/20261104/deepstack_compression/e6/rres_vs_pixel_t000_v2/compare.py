import argparse,collections,hashlib,json,math,os
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review';TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def terminal(name,reviewer):
 mon=I/name;s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or [(cwd/x).resolve() for x in s['command'] if Path(x).name==reviewer.name]!=[reviewer.resolve()]:raise ValueError('review supervisor')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args();out=args.out.resolve()
 if out!=(D/'result001.json').resolve() or out.exists():raise ValueError('output')
 plan=read(D/'plan001.json')
 if np.__version__!='2.2.6' or plan.get('status')!='frozen' or plan.get('GT_opened') is not False or any(sha(k)!=v for k,v in plan['source_sha256'].items()):raise ValueError('plan or sources')
 for n in ('pixel_dev384_result','pixel_dev384_scores','rres_k256_dev384_result','rres_k256_dev384_scores'):terminal('monitor_'+n,I/('review_'+n+'.py'))
 pf=read(B/'e6/pixelprune_dev384_accept/verification001.json');pr=read(I/'pixel_dev384_result_review.json');ps=read(I/'pixel_dev384_scores_review.json');rf=read(B/'e6/rres_k256_dev384/verification001.json');rr=read(I/'rres_k256_dev384_result_review.json');rs=read(I/'rres_k256_dev384_scores_review.json')
 if any(x.get('status')!='passed' or x.get('errors')!=[] for x in (pf,pr,ps,rf,rr,rs)) or any(x.get('independent_review') is not True for x in (pr,ps,rr,rs)) or pr.get('formal_sha256')!=sha(B/'e6/pixelprune_dev384_accept/verification001.json') or rr.get('formal_sha256')!=sha(B/'e6/rres_k256_dev384/verification001.json') or pr.get('questions')!=384 or rr.get('images')!=384 or ps.get('questions')!=384 or rs.get('questions')!=384:raise ValueError('acceptance')
 pp=B/'e6/pixelprune_dev384_analysis/score001/scores.jsonl';rp=B/'e6/rres_k256_full_analysis_v2/score001/scores.jsonl'
 if ps.get('score_sha256')!=sha(pp) or rs.get('score_sha256')!=sha(rp) or rr.get('files_sha256')!=rf.get('files_sha256'):raise ValueError('score proof identity')
 key=lambda x:(x['task'],str(x['qid']),x['group_id']);unique=lambda rows:{key(x):x for x in rows}
 pa=read(B/'e6/pixelprune_dev384/plan001.json')['anchors'];ra=read(B/'e6/rres_k256_dev384/design001.json')['anchors'];pm=unique(pa);rm=unique(ra);pv=lines(pp);rv=lines(rp);p=unique(pv);r=unique(rv)
 if any(len(x)!=384 for x in (pa,ra,pm,rm,pv,rv,p,r)) or not(set(pm)==set(rm)==set(p)==set(r)) or collections.Counter(k[0] for k in p)!={x:128 for x in TASKS}:raise ValueError('paired inventory')
 allrows=[]
 for k in pm:
  a,b=pm[k],rm[k]
  if any(a[x]!=b[x] for x in ('image_sha256','message_sha256')):raise ValueError('anchor input')
  name=k[0]+'_'+k[1]+'.json';px=B/'e6/pixelprune_dev384/run_smoke001/observations'/name;rx=B/'e6/rres_k256_dev384/run001/observations'/name
  if any(str(f) not in formal['files_sha256'] or formal['files_sha256'][str(f)]!=review.get(str(f)) or sha(f)!=formal['files_sha256'][str(f)] for f,formal,review in ((px,pf,pr['per_question_files_sha256']),(rx,rf,rr['files_sha256']))):raise ValueError('record acceptance')
  x,y=read(px),read(rx)
  if key(x)!=k or key(y)!=k or any(x[n]!=a[n] or y[n]!=a[n] for n in ('image_sha256','message_sha256')) or any(x['generation_kwargs'].get(n)!=v for n,v in y['generation_scalars'].items()) or x['eos_token_ids']!=y['eos_ids'] or x['generation_kwargs'].get('num_beams')!=1 or x['generation_kwargs'].get('use_cache') is not True:raise ValueError('identity or recorded controls')
  ps0=p[k]['arms']['pixel_dynamic'];rs0=r[k]['rres']
  if ps0['answer']!=x['decoded'] or ps0['eos']!=x['ends_eos'] or rs0['answer']!=y['answer'] or rs0['eos']!=y['eos'] or rs0['truncated']!=y['truncated']:raise ValueError('score to record')
  for score in (ps0,rs0):
   if not math.isfinite(score['score']) or not 0<=score['score']<=1 or abs(score['score']-score['independent_score'])>1e-12:raise ValueError('score validity')
  allrows.append({'task':k[0],'qid':k[1],'group_id':k[2],'rres':rs0['score'],'pixel':ps0['score'],'rres_eos':rs0['eos'],'pixel_eos':ps0['eos'],'rres_truncated':rs0['truncated'],'pixel_truncated':ps0['truncated']})
 reports={}
 for task in TASKS:
  rows=[x for x in allrows if x['task']==task];groups=collections.defaultdict(list)
  for x in rows:groups[x['group_id']].append((x['rres'],x['pixel']))
  ids=sorted(groups);sums=np.array([np.array(groups[g]).sum(axis=0) for g in ids]);sizes=np.array([len(groups[g]) for g in ids]);means=100*sums.sum(axis=0)/sizes.sum();draw=np.random.default_rng(20261104).integers(0,len(ids),(10000,len(ids)));boot=100*sums[draw].sum(axis=1)/sizes[draw].sum(axis=1)[:,None];diff=[x['rres']-x['pixel'] for x in rows]
  reports[task]={'questions':len(rows),'image_groups':len(ids),'rres_mean_pct':float(means[0]),'pixel_mean_pct':float(means[1]),'difference_pp':float(means[0]-means[1]),'ci95_pp':[float(v) for v in np.quantile(boot[:,0]-boot[:,1],[.025,.975])],'improved':sum(x>0 for x in diff),'decreased':sum(x<0 for x in diff),'unchanged':sum(x==0 for x in diff),'rres_eos':sum(x['rres_eos'] for x in rows),'pixel_eos':sum(x['pixel_eos'] for x in rows),'rres_truncated':sum(x['rres_truncated'] for x in rows),'pixel_truncated':sum(x['pixel_truncated'] for x in rows)}
 if any(sha(k)!=v for k,v in plan['source_sha256'].items()):raise ValueError('source drift')
 result={'status':'passed','questions':384,'GT_opened':False,'existing_predictions_opened':True,'plan_sha256':sha(D/'plan001.json'),'tasks':reports,'limitations':['Posthoc same DEV; uncorrected intervals.','R-res six recorded controls were inherited from E1 and checked online against actual kwargs and EOS, but actual per-request kwargs were not separately archived; Pixel recorded num_beams and use_cache.','Different resolution and dynamic visual budget; no equal-compute or speed claim.']}
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
