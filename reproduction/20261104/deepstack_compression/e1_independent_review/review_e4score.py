"""Independent E4 DEV score journal/statistics audit; does not open GT."""
import hashlib,json,math,os,time
from pathlib import Path
import numpy as np
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');D=B/'e4_analysis';R=B/'e1_independent_review'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 b=Path(p).read_bytes()
 if b and not b.endswith(b'\n'):raise ValueError('incomplete journal')
 return [json.loads(x) for x in b.splitlines()]
def main():
 started=time.time();errors=[];checks=0
 def req(v,m):
  nonlocal checks
  checks+=1
  if not bool(v):errors.append(m)
 try:
  run=D/'score_panel003';analysis=read(run/'analysis.json');frozen=read(run/'frozen.json');plan=read(D/'plan.json');term=read(D/'monitor_score_panel003/terminal.json');start=read(D/'monitor_score_panel003/started.json')
  req(term['reason']=='completed' and term['worker_returncode']==0 and term['remaining_process_group']==[] and term['worker_pid']==start['worker_pid'],'actual scorer exit')
  cmd=start['command'];cwd=Path(start['cwd']);req([(cwd/x).resolve() for x in cmd if Path(x).name=='score_e4.py']==[(D/'score_e4.py').resolve()] and cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==run,'score command')
  req(analysis['status']=='completed' and analysis['n_questions']==384 and analysis['n_predictions']==3072 and analysis['scores_sha256']==sha(run/'scores.jsonl') and analysis['plan_sha256']==sha(D/'plan.json'),'analysis and records binding')
  req(frozen['panel_proof_sha256']==sha(B/'e4_panel_v2/panel_003_verification.json') and frozen['review_proof_sha256']==sha(R/'e4panel003_review.json') and frozen['plan_sha256']==sha(D/'plan.json'),'accepted panel binding')
  for p,d in frozen['source_code_sha256'].items():req(sha(p)==d,'source '+p)
  for p,d in ((B/'e4_panel_v2/panel_003_verification.json','panel_proof_sha256'),(R/'e4panel003_review.json','review_proof_sha256'),(R/'monitor_e4panel003/started.json','review_started_sha256'),(R/'monitor_e4panel003/terminal.json','review_terminal_sha256')):req(sha(p)==frozen[d],'evidence '+d)
  e1=read(B/'e1/e1full001_completion.json');inputs=lines(e1['files']['inputs']['path']);scores=lines(run/'scores.jsonl');req(len(inputs)==len(scores)==384,'score denominator')
  req([(x['task'],str(x['qid']),x['group_id']) for x in inputs]==[(x['task'],str(x['qid']),x['group_id']) for x in scores],'DEV identity/order')
  configs=plan['configs'];tasks=plan['tasks'];counts={}
  for task in tasks:
   rows=[x for x in scores if x['task']==task];r=analysis['tasks'][task];req(len(rows)==128 and len({x['group_id'] for x in rows})==128,task+' group denominator')
   vals=[]
   for x in rows:
    req(set(x['arms'])==set(configs),task+' arm inventory');v=[]
    for arm in configs:
     z=x['arms'][arm];value=z['score'];other=z['independent_score']
     req(isinstance(value,(float,int)) and math.isfinite(value) and 0<=value<=1 and abs(value-other)<=1e-12,task+' author-independent score agreement');v.append(value)
    vals.append(v)
   matrix=np.asarray(vals,dtype=np.float64);ordered=sorted(range(len(rows)),key=lambda i:rows[i]['group_id']);grouped=matrix[ordered]
   rng=np.random.default_rng(plan['bootstrap']['seed']);draw=rng.integers(0,len(rows),(plan['bootstrap']['replicates'],len(rows)));boot=grouped[draw].mean(axis=1)*100
   for i,arm in enumerate(configs):
    req(abs(r['arm_means_pct'][arm]-100*matrix[:,i].mean())<1e-9,task+' arm mean '+arm)
    req(np.allclose(r['arm_percentile_95_ci_pct'][arm],np.quantile(boot[:,i],[.025,.975]),rtol=0,atol=1e-9),task+' arm CI '+arm)
   for name,(a,b) in plan['contrasts'].items():
    i=configs.index(a);j=configs.index(b);d=matrix[:,i]-matrix[:,j];result=r['contrasts'][name];ci=np.quantile(boot[:,i]-boot[:,j],[.025,.975])
    req(abs(result['difference_pp']-100*d.mean())<1e-9 and np.allclose(result['image_cluster_percentile_95_ci_pp'],ci,rtol=0,atol=1e-9),task+' paired contrast '+name)
    req(result['improved']==int((d>1e-12).sum()) and result['declined']==int((d< -1e-12).sum()) and result['unchanged']==int((abs(d)<=1e-12).sum()) and result['answer_changed']==sum(x['arms'][a]['answer']!=x['arms'][b]['answer'] for x in rows),task+' change counts '+name)
   counts[task]=len(rows)
 except Exception as exc:
  import traceback;traceback.print_exc();errors.append('exception '+repr(exc));counts={}
 report={'status':'passed' if not errors and counts=={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128} else 'failed','questions_verified':sum(counts.values()),'arms_verified':sum(counts.values())*8,'checks':checks,'errors':errors,'score_journal_sha256':sha(D/'score_panel003/scores.jsonl'),'analysis_sha256':sha(D/'score_panel003/analysis.json'),'score_terminal_sha256':sha(D/'monitor_score_panel003/terminal.json'),'reviewer_sha256':sha(__file__),'GT_opened':False,'scope':'independent journal and statistics reproduction; item-level scores checked against stored independent scorer, GT not reopened','start_unix':started,'end_unix':time.time()}
 out=R/'e4score003_review.json'
 with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(report));return 0 if report['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
