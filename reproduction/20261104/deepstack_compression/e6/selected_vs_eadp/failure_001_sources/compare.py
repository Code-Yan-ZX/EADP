"""Post-hoc paired DEV comparison; consumes only separately accepted artifacts."""
import argparse,collections,hashlib,json,math,os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PLAN='c9076484a343c2d75e28bcffbd2879060281e926796a30811412f46b0a01e152'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def key(r):return (r['task'],str(r['qid']),r['group_id'])
def unique(rows,fn):
 d={}
 for r in rows:
  k=fn(r)
  if k in d:raise ValueError('duplicate matched record')
  d[k]=r
 return d
def supervised(mon,script):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if not (t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==s['process_group']==t['worker_pid'] and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()]):raise ValueError('unverified terminal')
def require_proof(proof,expected):
 if any(proof.get(k)!=v for k,v in expected.items()):raise ValueError('upstream proof changed or failed')
def require_pair(r,er,inp,score):
 if not (r['image_sha256']==inp['image_sha256'] and r['message_sha256']==er['message_sha256']==inp['message_sha256'] and r['processor']==er['processor'] and r['main_visual']==er['visual_sha256'] and r['generation_scalars']==er['generation_scalars'] and r['eos_ids']==er['eos_ids'] and len(r['spatial_ids'])==len(er['spatial_ids'])==256 and score['score']==score['independent_score']):raise ValueError('paired identity or scoring mismatch')
def analyze(rows,seed=20261104,replicates=10000):
 if not rows:raise ValueError('empty comparison')
 groups=collections.defaultdict(list)
 for r in rows:
  if not r['group_id'] or any(not math.isfinite(r[x]) or not 0<=r[x]<=1 for x in ('eadp','div','cd')):raise ValueError('bad score or group')
  groups[r['group_id']].append(r)
 names=sorted(groups);sums=np.array([[sum(r[x] for r in groups[g]) for x in ('eadp','div','cd')] for g in names]);sizes=np.array([len(groups[g]) for g in names]);rng=np.random.default_rng(seed);draw=rng.integers(0,len(names),(replicates,len(names)));boot=sums[draw].sum(axis=1)/sizes[draw].sum(axis=1)[:,None]
 result={'n_questions':len(rows),'n_groups':len(names),'contrasts':{}}
 for i,name in ((1,'div-adapted_minus_eadp'),(2,'cd-adapted_minus_eadp')):
  delta=np.array([r['div' if i==1 else 'cd']-r['eadp'] for r in rows]);ci=np.quantile((boot[:,i]-boot[:,0])*100,[.025,.975]);result['contrasts'][name]={'difference_pp':float(delta.mean()*100),'percentile_95_ci_pp':[float(x) for x in ci],'improved':int((delta>1e-12).sum()),'declined':int((delta< -1e-12).sum()),'unchanged':int((abs(delta)<=1e-12).sum())}
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve()
 if out!=(HERE/'result_001.json').resolve() or sha(HERE/'plan_003.json')!=PLAN:raise ValueError('plan or output')
 plan=read(HERE/'plan_003.json')
 if sha(HERE/'plan.json')!=plan['prior_plan_sha256'] or sha(HERE/'plan_002.json')!=plan['revision_002_sha256'] or np.__version__!=plan['numpy_version']:raise ValueError('prior plan or NumPy version changed')
 cpu=read(HERE/'cpu_check_005/verification.json');review=read(HERE/'independent_prelaunch_review.json')
 supervised(HERE/'monitor_cpu005',HERE/'cpu_validation.py');supervised(ROOT/'e1_independent_review/monitor_selected_vs_eadp_prelaunch',ROOT/'e1_independent_review/review_selected_vs_eadp.py')
 required={str((HERE/n).resolve()) for n in ('plan.json','plan_002.json','plan_003.json','compare.py','test_compare.py','cpu_validation.py')}
 if not (cpu['status']=='passed' and cpu['source_sha256_before']==cpu['source_sha256_after']==review['source_sha256'] and set(review['source_sha256'])==required and all(sha(p)==h for p,h in review['source_sha256'].items()) and review['status']=='passed' and review['independent_review'] is True and review['reviewer_sha256']==sha(ROOT/'e1_independent_review/review_selected_vs_eadp.py') and review['cpu_sha256']==sha(HERE/'cpu_check_005/verification.json') and review['cpu_started_sha256']==sha(HERE/'monitor_cpu005/started.json') and review['cpu_terminal_sha256']==sha(HERE/'monitor_cpu005/terminal.json')):raise ValueError('comparison source review missing')
 if any(sha(p)!=v for p,v in plan['inputs_sha256'].items()):raise ValueError('frozen input changed')
 for mon,script in [('e1_independent_review/monitor_e1full001','e1_independent_review/review_full.py'),('e1_independent_review/monitor_e1score001','e1_independent_review/review_score.py'),('e1_independent_review/monitor_selected_full_panel001','e1_independent_review/review_selected_full_panel.py'),('e1_independent_review/monitor_e6_selected_scores001','e1_independent_review/review_e6_selected_scores.py')]:supervised(ROOT/mon,ROOT/script)
 for mon,script in [('e1/monitorverifye1full001','e1_analysis/completion.py'),('e1/monitorscoree1full001','e1_analysis/score.py'),('e6/selected_full/monitor_verify001','e6/selected_full/verify.py'),('e6/selected_full/monitor_panel001','e6/selected_full/launcher.py'),('e6/selected_analysis/monitor_score002','e6/selected_analysis/score_selected.py')]:supervised(ROOT/mon,ROOT/script)
 ep=read(ROOT/'e1/e1full001_completion.json');ei=read(ROOT/'e1_independent_review/e1full001_review.json');sp=read(ROOT/'e6/selected_full/panel_001_verification.json');si=read(ROOT/'e1_independent_review/selected_full_panel001_review.json')
 require_proof(ep,{'status':'passed','stage':'full','natural_questions':384,'natural_records':1536,'GT_opened':False,'scoring_performed':False,'verifier_sha256':sha(ROOT/'e1_analysis/completion.py')})
 require_proof(ei,{'status':'passed','n_verified':384,'errors':[],'reviewer_sha256':sha(ROOT/'e1_independent_review/review_full.py'),'completion_proof_sha256':sha(ROOT/'e1/e1full001_completion.json'),'completion_started_sha256':sha(ROOT/'e1/monitorverifye1full001/started.json'),'completion_terminal_sha256':sha(ROOT/'e1/monitorverifye1full001/terminal.json')})
 require_proof(sp,{'status':'passed','stage':'full_dev','questions':384,'generations':1536,'GT_opened':False,'scoring':False,'verifier_sha256':sha(ROOT/'e6/selected_full/verify.py')})
 require_proof(si,{'status':'passed','independent_review':True,'questions':384,'generations':1536,'errors':[],'reviewer_sha256':sha(ROOT/'e1_independent_review/review_selected_full_panel.py'),'formal_proof_sha256':sha(ROOT/'e6/selected_full/panel_001_verification.json'),'formal_started_sha256':sha(ROOT/'e6/selected_full/monitor_verify001/started.json'),'formal_terminal_sha256':sha(ROOT/'e6/selected_full/monitor_verify001/terminal.json'),'run_started_sha256':sha(ROOT/'e6/selected_full/monitor_panel001/started.json'),'run_terminal_sha256':sha(ROOT/'e6/selected_full/monitor_panel001/terminal.json')})
 for name in ('journal','inputs'):
  if sha(ep['files'][name]['path'])!=ep['files'][name]['sha256']:raise ValueError('E1 accepted input changed')
 selected_records=ROOT/'e6/selected_full/panel_001/records.jsonl'
 if sp['files_sha256'].get(str(selected_records))!=sha(selected_records):raise ValueError('selected accepted records changed')
 eproof=read(ROOT/'e1_independent_review/e1full001_score_review.json');sproof=read(ROOT/'e1_independent_review/e6_selected_scores_review.json')
 require_proof(eproof,{'status':'passed','n_verified':384,'arms_verified':1536,'errors':[],'score_sha256':sha(ROOT/'e1/e1full001_score/scores.jsonl'),'e1_review_sha256':sha(ROOT/'e1_independent_review/e1full001_review.json'),'score_terminal_sha256':sha(ROOT/'e1/monitorscoree1full001/terminal.json'),'reviewer_sha256':sha(ROOT/'e1_independent_review/review_score.py')})
 require_proof(sproof,{'status':'passed','independent_review':True,'questions':384,'scores_verified':1536,'CONFIRM_FINAL_answers_parsed':False,'scores_sha256':sha(ROOT/'e6/selected_analysis/score002/scores.jsonl'),'formal_sha256':sha(ROOT/'e6/selected_full/panel_001_verification.json'),'panel_review_sha256':sha(ROOT/'e1_independent_review/selected_full_panel001_review.json'),'score_started_sha256':sha(ROOT/'e6/selected_analysis/monitor_score002/started.json'),'score_terminal_sha256':sha(ROOT/'e6/selected_analysis/monitor_score002/terminal.json'),'reviewer_sha256':sha(ROOT/'e1_independent_review/review_e6_selected_scores.py')})
 e=unique([x for x in lines(ROOT/'e1/e1full001/raw_predictions.jsonl') if x['arm']=='P3-D1' and x['forced'] is False],key);inp=unique(lines(ROOT/'e1/input_preparation_002/inputs.jsonl'),key);x=unique([r for r in lines(ROOT/'e6/selected_full/panel_001/records.jsonl') if r['path']=='adapted'],lambda r:(r['method'],*key(r)));es=unique(lines(ROOT/'e1/e1full001_score/scores.jsonl'),key);ss=unique(lines(ROOT/'e6/selected_analysis/score002/scores.jsonl'),key)
 if not (len(e)==len(inp)==len(es)==len(ss)==384 and len(x)==768 and set(e)==set(inp)==set(es)==set(ss) and set(x)=={(m,*k) for k in e for m in ('div','cd')}):raise ValueError('unmatched panel inventory')
 paired=[]
 for k,er in e.items():
  row={'task':k[0],'group_id':k[2]};sc=es[k]['arms']['P3-D1'];row['eadp']=float(sc['score'])
  if sc['score']!=sc['independent_score'] or sc['raw_answer']!=er['answer'] or sc['eos']!=er['eos'] or sc['truncated']!=er['truncated']:raise ValueError('EADP score disagreement')
  for method in ('div','cd'):
   r=x[(method,*k)];s=ss[k]['arms'][method+':adapted']
   require_pair(r,er,inp[k],s)
   if s['answer']!=r['answer'] or s['eos']!=r['eos'] or s['truncated']!=r['truncated']:raise ValueError('selected score record mismatch')
   row[method]=float(s['score'])
  paired.append(row)
 if collections.Counter(r['task'] for r in paired)!={'DocVQA_VAL':128,'TextVQA_VAL':128,'AI2D_TEST':128}:raise ValueError('task denominator')
 reports={task:analyze(paired if task=='ALL' else [r for r in paired if r['task']==task]) for task in plan['strata']}
 data={'status':'completed','plan_sha256':PLAN,'source_sha256':sha(__file__),'reports':reports,'scope':plan['post_hoc_disclosure'],'limitations':plan['constraints']}
 with out.open('x') as f:json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
