import argparse,collections,hashlib,json,math,os
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent;B=D.parents[1];I=B/'e1_independent_review';TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
DEADLINE=datetime(2026,11,7,15,30,14,tzinfo=timezone.utc)
CONTROLS={'do_sample':False,'max_new_tokens':2048,'top_p':0.001,'top_k':1,'temperature':0.01,'repetition_penalty':1.0}
def check_deadline(now):
 if now.tzinfo is None or now>=DEADLINE:raise ValueError('deadline')
def controls(x,y):
 inherited=y.get('generation_scalars');kw=x.get('generation_kwargs',{})
 if not isinstance(inherited,dict) or set(inherited)!=set(CONTROLS) or any(type(inherited[k]) is not type(v) or inherited[k]!=v or type(kw.get(k)) is not type(v) or kw.get(k)!=v for k,v in CONTROLS.items()) or x.get('eos_token_ids')!=[151645,151643] or y.get('eos_ids')!=[151645,151643] or type(kw.get('num_beams')) is not int or kw['num_beams']!=1 or kw.get('use_cache') is not True:raise ValueError('control mismatch')
def check_score(score):
 if not math.isfinite(score['score']) or not math.isfinite(score['independent_score']) or not 0<=score['score']<=1 or abs(score['score']-score['independent_score'])>1e-12:raise ValueError('score validity')
def terminal(name,reviewer,out=None):
 mon=I/name;s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd'])
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or [(cwd/x).resolve() for x in s['command'] if Path(x).name==reviewer.name]!=[reviewer.resolve()] or (out is not None and (s['command'].count('--out')!=1 or (cwd/s['command'][s['command'].index('--out')+1]).resolve()!=out.resolve())):raise ValueError('review supervisor')
def require_prelaunch(plan):
 pre=I/'rres_vs_pixel_t020_prelaunch_review.json';reviewer=I/'review_rres_vs_pixel_t020_prelaunch.py';terminal('monitor_rres_vs_pixel_t020_prelaunch',reviewer,pre);q=read(pre);cpu=read(D/'cpu_proof001.json');cs=read(D/'monitor_cpu001/started.json');ct=read(D/'monitor_cpu001/terminal.json');cwd=Path(cs['cwd'])
 if ct.get('reason')!='completed' or ct.get('worker_returncode')!=0 or ct.get('remaining_process_group')!=[] or cs.get('worker_pid')!=ct.get('worker_pid') or [(cwd/x).resolve() for x in cs['command'] if Path(x).name=='cpu_check.py']!=[(D/'cpu_check.py').resolve()]:raise ValueError('CPU supervisor')
 if q.get('status')!='passed' or q.get('independent_review') is not True or q.get('GT_opened') is not False or q.get('scores_parsed') is not False or q.get('errors')!=[] or q.get('plan_sha256')!=sha(D/'plan001.json') or q.get('cpu_proof_sha256')!=sha(D/'cpu_proof001.json') or q.get('reviewer_sha256')!=sha(reviewer) or q.get('cpu_started_sha256')!=sha(D/'monitor_cpu001/started.json') or q.get('cpu_terminal_sha256')!=sha(D/'monitor_cpu001/terminal.json') or cpu.get('status')!='passed' or cpu.get('GT_opened') is not False or cpu.get('scores_parsed') is not False or cpu.get('plan_sha256')!=sha(D/'plan001.json') or cpu.get('source_sha256')!=plan['source_sha256']:raise ValueError('prelaunch proof')
def summarize(rows):
 groups=collections.defaultdict(list)
 for x in rows:groups[x['group_id']].append((x['rres'],x['pixel']))
 ids=sorted(groups);sums=np.array([np.array(groups[g]).sum(axis=0) for g in ids]);sizes=np.array([len(groups[g]) for g in ids]);means=100*sums.sum(axis=0)/sizes.sum();draw=np.random.default_rng(20261104).integers(0,len(ids),(10000,len(ids)));boot=100*sums[draw].sum(axis=1)/sizes[draw].sum(axis=1)[:,None];diff=[x['rres']-x['pixel'] for x in rows]
 return {'questions':len(rows),'image_groups':len(ids),'rres_mean_pct':float(means[0]),'pixel_mean_pct':float(means[1]),'difference_pp':float(means[0]-means[1]),'ci95_pp':[float(v) for v in np.quantile(boot[:,0]-boot[:,1],[.025,.975])],'improved':sum(x>0 for x in diff),'decreased':sum(x<0 for x in diff),'unchanged':sum(x==0 for x in diff),'rres_eos':sum(x['rres_eos'] for x in rows),'pixel_eos':sum(x['pixel_eos'] for x in rows),'rres_truncated':sum(x['rres_truncated'] for x in rows),'pixel_truncated':sum(x['pixel_truncated'] for x in rows)}
def main():
 check_deadline(datetime.now(timezone.utc))
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path);args=ap.parse_args();out=args.out.resolve()
 if out!=(D/'result001.json').resolve() or out.exists():raise ValueError('output')
 plan=read(D/'plan001.json')
 if datetime.fromtimestamp(read(B/'initial_plan.json')['budget_end_unix'],timezone.utc)!=DEADLINE:raise ValueError('deadline plan mismatch')
 if np.__version__!='2.2.6' or plan.get('status')!='frozen' or plan.get('GT_opened') is not False or any(sha(k)!=v for k,v in plan['source_sha256'].items()):raise ValueError('plan or sources')
 require_prelaunch(plan)
 for n in ('pixel_t020_dev384_result','pixel_t020_dev384_scores','rres_k256_dev384_result','rres_k256_dev384_scores'):terminal('monitor_'+n,I/('review_'+n+'.py'))
 for mon,script,outdir in ((B/'e6/pixelprune_t020_dev384_analysis/monitor_score001',B/'e6/pixelprune_t020_dev384_analysis/score.py',B/'e6/pixelprune_t020_dev384_analysis/score001'),(B/'e6/rres_k256_full_analysis_v2/monitor_score001',B/'e6/rres_k256_full_analysis_v2/score.py',B/'e6/rres_k256_full_analysis_v2/score001')):
  z=read(mon/'started.json');t=read(mon/'terminal.json');cmd=z['command'];cwd=Path(z['cwd'])
  if z.get('worker_pid')!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=outdir.resolve():raise ValueError('scoring supervisor identity')
 pf=read(B/'e6/pixelprune_t020_accept/verification001.json');pr=read(I/'pixel_t020_dev384_result_review.json');ps=read(I/'pixel_t020_dev384_scores_review.json');rf=read(B/'e6/rres_k256_dev384/verification001.json');rr=read(I/'rres_k256_dev384_result_review.json');rs=read(I/'rres_k256_dev384_scores_review.json')
 if any(x.get('status')!='passed' or x.get('errors')!=[] for x in (pf,pr,ps,rf,rr,rs)) or any(x.get('independent_review') is not True for x in (pr,ps,rr,rs)) or pr.get('formal_sha256')!=sha(B/'e6/pixelprune_t020_accept/verification001.json') or rr.get('formal_sha256')!=sha(B/'e6/rres_k256_dev384/verification001.json') or pr.get('questions')!=384 or rr.get('images')!=384 or ps.get('questions')!=384 or rs.get('questions')!=384:raise ValueError('acceptance')
 pp=B/'e6/pixelprune_t020_dev384_analysis/score001/scores.jsonl';rp=B/'e6/rres_k256_full_analysis_v2/score001/scores.jsonl'
 for proof,name,mon in ((pr,'pixel_t020_dev384_result','monitor_pixel_t020_dev384_result'),(ps,'pixel_t020_dev384_scores','monitor_pixel_t020_dev384_scores'),(rr,'rres_k256_dev384_result','monitor_rres_k256_dev384_result'),(rs,'rres_k256_dev384_scores','monitor_rres_k256_dev384_scores')):
  if proof.get('reviewer_sha256')!=sha(I/('review_'+name+'.py')):raise ValueError('reviewer source')
 for proof,mon in ((ps,B/'e6/pixelprune_t020_dev384_analysis/monitor_score001'),(rs,B/'e6/rres_k256_full_analysis_v2/monitor_score001')):
  if proof.get('score_started_sha256')!=sha(mon/'started.json') or proof.get('score_terminal_sha256')!=sha(mon/'terminal.json'):raise ValueError('scoring supervisor binding')
 if ps.get('score_sha256')!=sha(pp) or rs.get('score_sha256')!=sha(rp) or rr.get('files_sha256')!=rf.get('files_sha256'):raise ValueError('score proof identity')
 key=lambda x:(x['task'],str(x['qid']),x['group_id']);unique=lambda rows:{key(x):x for x in rows}
 pa=read(B/'e6/pixelprune_t020_dev384/plan001.json')['anchors'];ra=read(B/'e6/rres_k256_dev384/design001.json')['anchors'];pm=unique(pa);rm=unique(ra);pv=lines(pp);rv=lines(rp);p=unique(pv);r=unique(rv)
 if any(len(x)!=384 for x in (pa,ra,pm,rm,pv,rv,p,r)) or not(set(pm)==set(rm)==set(p)==set(r)) or collections.Counter(k[0] for k in p)!={x:128 for x in TASKS}:raise ValueError('paired inventory')
 allrows=[]
 for k in pm:
  a,b=pm[k],rm[k]
  if any(a[x]!=b[x] for x in ('image_sha256','message_sha256')):raise ValueError('anchor input')
  name=k[0]+'_'+k[1]+'.json';px=B/'e6/pixelprune_t020_dev384/run_smoke001/observations'/name;rx=B/'e6/rres_k256_dev384/run001/observations'/name
  if any(str(f) not in formal['files_sha256'] or formal['files_sha256'][str(f)]!=review.get(str(f)) or sha(f)!=formal['files_sha256'][str(f)] for f,formal,review in ((px,pf,pr['per_question_files_sha256']),(rx,rf,rr['files_sha256']))):raise ValueError('record acceptance')
  x,y=read(px),read(rx)
  if key(x)!=k or key(y)!=k or any(x[n]!=a[n] or y[n]!=a[n] for n in ('image_sha256','message_sha256')) :raise ValueError('identity or recorded controls')
  controls(x,y)
  ps0=p[k]['arms']['pixel_t020'];rs0=r[k]['rres']
  if ps0['answer']!=x['decoded'] or ps0['eos']!=x['ends_eos'] or rs0['answer']!=y['answer'] or rs0['eos']!=y['eos'] or rs0['truncated']!=y['truncated']:raise ValueError('score to record')
  for score in (ps0,rs0):check_score(score)
  allrows.append({'task':k[0],'qid':k[1],'group_id':k[2],'rres':rs0['score'],'pixel':ps0['score'],'rres_eos':rs0['eos'],'pixel_eos':ps0['eos'],'rres_truncated':rs0['truncated'],'pixel_truncated':ps0['truncated']})
 reports={}
 for task in TASKS:
  rows=[x for x in allrows if x['task']==task];reports[task]=summarize(rows)
 if any(sha(k)!=v for k,v in plan['source_sha256'].items()):raise ValueError('source drift')
 result={'status':'passed','questions':384,'GT_opened':False,'existing_predictions_opened':True,'plan_sha256':sha(D/'plan001.json'),'tasks':reports,'limitations':['Posthoc same DEV; uncorrected intervals.','R-res six recorded controls were inherited from E1 and checked online against actual kwargs and EOS, but actual per-request kwargs were not separately archived; Pixel recorded num_beams and use_cache.','Different resolution and dynamic visual budget; no equal-compute or speed claim.']}
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
