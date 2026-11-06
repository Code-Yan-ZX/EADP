"""Post-run independent tau=.2 paired/bootstrap recomputation; no truth files."""
import argparse,collections,hashlib,json,math,os
from pathlib import Path
import numpy as np
I=Path(__file__).resolve().parent;B=I.parent;D=B/'e6/rres_vs_pixel_t020'
TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def clean(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or s.get('worker_pid')!=t.get('worker_pid') or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()] or (out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve())):raise ValueError('supervisor '+str(mon))
def keyed(rows):
 out={}
 for r in rows:
  k=(r['task'],str(r['qid']),r['group_id'])
  if k in out:raise ValueError('duplicate question')
  out[k]=r
 return out
def close(x,y):return math.isclose(float(x),float(y),rel_tol=0,abs_tol=1e-10)
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();dest=I/'rres_vs_pixel_t020_review.json'
 if a.out.resolve()!=dest or dest.exists() or np.__version__!='2.2.6':raise ValueError('output or NumPy')
 plan=read(D/'plan001.json');res=read(D/'result001.json');clean(D/'monitor_compare001',D/'compare.py',D/'result001.json');clean(D/'monitor_freeze001',D/'freeze.py')
 pre=I/'rres_vs_pixel_t020_prelaunch_review.json';prereviewer=I/'review_rres_vs_pixel_t020_prelaunch.py';clean(I/'monitor_rres_vs_pixel_t020_prelaunch',prereviewer,pre);prior=read(pre)
 if prior.get('status')!='passed' or prior.get('independent_review') is not True or prior.get('plan_sha256')!=sha(D/'plan001.json') or prior.get('reviewer_sha256')!=sha(prereviewer) or prior.get('scores_parsed') is not False:raise ValueError('prelaunch')
 if plan['status']!='frozen' or plan['GT_opened'] is not False or res['status']!='passed' or res['questions']!=384 or res['GT_opened'] is not False or res['plan_sha256']!=sha(D/'plan001.json') or any(sha(k)!=v for k,v in plan['source_sha256'].items()):raise ValueError('comparison plan or result')
 specs=[('pixel_t020_dev384_result','questions','e6/pixelprune_t020_accept/verification001.json'),('rres_k256_dev384_result','images','e6/rres_k256_dev384/verification001.json'),('pixel_t020_dev384_scores','questions',None),('rres_k256_dev384_scores','questions',None)];proofs={}
 for name,count,formal in specs:
  q=I/(name+'_review.json');reviewer=I/('review_'+name+'.py');clean(I/('monitor_'+name),reviewer,q);v=read(q)
  if v['status']!='passed' or v['independent_review'] is not True or v[count]!=384 or v['errors']!=[] or v['reviewer_sha256']!=sha(reviewer) or (formal and v['formal_sha256']!=sha(B/formal)):raise ValueError('prior independent '+name)
  proofs[name]=v
 score_paths={'pixel':B/'e6/pixelprune_t020_dev384_analysis/score001/scores.jsonl','rres':B/'e6/rres_k256_full_analysis_v2/score001/scores.jsonl'}
 if any(proofs[n]['score_sha256']!=sha(score_paths[a]) for n,a in [('pixel_t020_dev384_scores','pixel'),('rres_k256_dev384_scores','rres')]):raise ValueError('scored sources')
 if proofs['rres_k256_dev384_scores']['result_review_sha256']!=sha(I/'rres_k256_dev384_result_review.json'):raise ValueError('rres score/result')
 pa=keyed(read(B/'e6/pixelprune_t020_dev384/plan001.json')['anchors']);ra=keyed(read(B/'e6/rres_k256_dev384/design001.json')['anchors']);ps=keyed(lines(score_paths['pixel']));rs=keyed(lines(score_paths['rres']))
 if not(len(pa)==len(ra)==len(ps)==len(rs)==384 and set(pa)==set(ra)==set(ps)==set(rs)) or collections.Counter(k[0] for k in pa)!={t:128 for t in TASKS}:raise ValueError('full paired inventory')
 pf=read(B/'e6/pixelprune_t020_accept/verification001.json');rf=read(B/'e6/rres_k256_dev384/verification001.json');pixel_review=proofs['pixel_t020_dev384_result'];rres_review=proofs['rres_k256_dev384_result'];output=[]
 for k in sorted(pa):
  a0=pa[k];b0=ra[k];name=k[0]+'_'+k[1]+'.json';px=B/'e6/pixelprune_t020_dev384/run_smoke001/observations'/name;rx=B/'e6/rres_k256_dev384/run001/observations'/name
  for file,formal,review in ((px,pf,pixel_review['per_question_files_sha256']),(rx,rf,rres_review['files_sha256'])):
   if sha(file)!=formal['files_sha256'].get(str(file)) or sha(file)!=review.get(str(file)):raise ValueError('record identity')
  x=read(px);y=read(rx)
  for name0 in ('image_sha256','message_sha256'):
   if not(a0[name0]==b0[name0]==x[name0]==y[name0]):raise ValueError('input identity')
  if (x['task'],str(x['qid']),x['group_id'])!=k or (y['task'],str(y['qid']),y['group_id'])!=k:raise ValueError('question identity')
  if set(y['generation_scalars'])!={'do_sample','max_new_tokens','top_p','top_k','temperature','repetition_penalty'} or any(x['generation_kwargs'].get(n)!=v for n,v in y['generation_scalars'].items()) or x['generation_kwargs'].get('num_beams')!=1 or x['generation_kwargs'].get('use_cache') is not True or x['eos_token_ids']!=y['eos_ids']:raise ValueError('generation controls')
  u=ps[k]['arms']['pixel_t020'];v=rs[k]['rres']
  if u['answer']!=x['decoded'] or u['eos']!=x['ends_eos'] or u['truncated'] is not (not x['ends_eos']) or v['answer']!=y['answer'] or v['eos']!=y['eos'] or v['truncated']!=y['truncated']:raise ValueError('score/actual output')
  for z in (u,v):
   if not(math.isfinite(z['score']) and 0<=z['score']<=1 and close(z['score'],z['independent_score'])):raise ValueError('score validity')
  output.append((k,u,v))
 for task in TASKS:
  rows=[x for x in output if x[0][0]==task]; groups=sorted({k[2] for k,_,_ in rows});g={z:[] for z in groups}
  for k,u,v in rows:g[k[2]].append((v['score'],u['score']))
  sums=np.array([[sum(z[0] for z in g[x]),sum(z[1] for z in g[x])] for x in groups]);sizes=np.array([len(g[x]) for x in groups]);rng=np.random.default_rng(20261104);draw=rng.integers(len(groups),size=(10000,len(groups)));bootdiff=100*((sums[:,0]-sums[:,1])[draw].sum(axis=1)/sizes[draw].sum(axis=1));lo,hi=np.quantile(bootdiff,[.025,.975]);r=read(D/'result001.json')['tasks'][task]
  truth={'questions':len(rows),'image_groups':len(groups),'rres_mean_pct':100*sum(v['score'] for _,_,v in rows)/len(rows),'pixel_mean_pct':100*sum(u['score'] for _,u,_ in rows)/len(rows),'difference_pp':100*sum(v['score']-u['score'] for _,u,v in rows)/len(rows),'improved':sum(v['score']>u['score'] for _,u,v in rows),'decreased':sum(v['score']<u['score'] for _,u,v in rows),'unchanged':sum(v['score']==u['score'] for _,u,v in rows),'rres_eos':sum(v['eos'] for _,u,v in rows),'pixel_eos':sum(u['eos'] for _,u,v in rows),'rres_truncated':sum(v['truncated'] for _,u,v in rows),'pixel_truncated':sum(u['truncated'] for _,u,v in rows)}
  if any(not close(r[k],v) for k,v in truth.items()) or not(close(r['ci95_pp'][0],lo) and close(r['ci95_pp'][1],hi)):raise ValueError('independent recomputation '+task)
 value={'status':'passed','independent_review':True,'questions':384,'GT_opened':False,'existing_predictions_opened':True,'errors':[],'plan_sha256':sha(D/'plan001.json'),'prelaunch_sha256':sha(pre),'result_sha256':sha(D/'result001.json'),'compare_started_sha256':sha(D/'monitor_compare001/started.json'),'compare_terminal_sha256':sha(D/'monitor_compare001/terminal.json'),'reviewer_sha256':sha(__file__),'limitations':['Additional posthoc comparison after seeing tau=0 and both full DEV outcomes.','R-res six generation kwargs and EOS were checked online against inherited E1 values; full effective generation config and per-request observed kwargs were not archived.','Different visual resolution and dynamic visual budget; no equal-compute or speed claim.','Uncorrected taskwise bootstrap intervals are exploratory.']}
 with dest.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
