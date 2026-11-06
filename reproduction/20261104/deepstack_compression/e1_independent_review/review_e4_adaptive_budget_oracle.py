"""Independently audit posthoc E4 DEV answer-informed budget optimum via MILP."""
import argparse,collections,hashlib,json,math,os
from decimal import Decimal
from pathlib import Path
import numpy as np,scipy
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix
R=Path(__file__).resolve().parent;B=R.parent;D=B/'e4_adaptive_budget_oracle';O=D/'result001'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def req(x,m):
 if not bool(x):raise ValueError(m)
def terminal(d):
 s=read(d/'started.json');t=read(d/'terminal.json');req(t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'],'supervisor');return s,t
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();plan=read(D/'plan001.json');s,t=terminal(D/'monitor_analysis001');report=read(O/'analysis.json');source=B/plan['source_scores'];cmd=s['command'];cwd=Path(s['cwd'])
 req([(cwd/x).resolve() for x in cmd if Path(x).name=='analyze.py']==[(D/'analyze.py').resolve()] and cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==O,'analysis command')
 req(report['status']=='completed' and report['plan_sha256']==sha(D/'plan001.json') and report['source_sha256']==sha(source)==plan['source_scores_sha256'] and report['GT_opened'] is False and report['new_predictions_opened'] is False,'source identity')
 first=read(B/plan['source_reviews'][0]);second=read(B/plan['source_reviews'][1]);req(first['status']=='passed' and first['questions_verified']==384 and first['score_journal_sha256']==sha(source) and second['status']=='passed' and second['questions']==384 and second['score_sha256']==sha(source),'accepted score reviews')
 terminal(R/'monitor_e4score003');terminal(B/'e4_analysis/independent_score_review/monitor001')
 with source.open() as h:rows=[json.loads(x,parse_float=Decimal) for x in h if x.strip()]
 with (O/'assignments.jsonl').open() as h:assign=[json.loads(x,parse_float=Decimal) for x in h if x.strip()]
 req(len(rows)==len(assign)==384 and len({(x['task'],str(x['qid']),x['group_id']) for x in rows})==384,'inventory')
 A={(x['task'],str(x['qid']),x['group_id']):x for x in assign};req(len(A)==384 and set(A)=={(x['task'],str(x['qid']),x['group_id']) for x in rows},'assignment inventory')
 tasks={}
 for task in plan['tasks']:
  part=[x for x in rows if x['task']==task];req(len(part)==128,'task denominator');v=np.empty((128,3));chosen=[];counts=collections.Counter();exact=Decimal(0);base=Decimal(0);used=0
  for i,r in enumerate(part):
   x=A[(task,str(r['qid']),r['group_id'])];k=x['selected_k'];req(k in (128,256,512),'selected budget');scores=[r['arms'][f'K{z}-D1']['score'] for z in (128,256,512)]
   req(all(z.is_finite() and 0<=z<=1 for z in scores) and all(abs(z-r['arms'][f'K{kk}-D1']['independent_score'])<=Decimal('1e-12') for kk,z in zip((128,256,512),scores)),'score')
   value=scores[(128,256,512).index(k)];req(abs(x['oracle_score']-value)<=Decimal('1e-14') and abs(x['fixed_k256_score']-scores[1])<=Decimal('1e-14'),'assignment scores')
   exact+=value;base+=scores[1];used+=k;counts[k]+=1;v[i]=[float(z) for z in scores];chosen.append(k)
  req(used<=32768 and exact>=base,'feasibility or baseline');C=lil_matrix((129,384),dtype=float)
  for i in range(128):C[i,3*i:3*i+3]=1;C[128,3*i:3*i+3]=[1,2,4]
  lb=np.r_[np.ones(128),-np.inf];ub=np.r_[np.ones(128),256.];res=milp(-v.ravel(),integrality=np.ones(384),bounds=Bounds(0,1),constraints=LinearConstraint(C.tocsr(),lb,ub),options={'time_limit':120,'mip_rel_gap':0.0})
  req(res.success and res.status==0 and res.x is not None,'MILP did not prove numeric optimum');other=float(-res.fun);observed=float(exact);req(abs(other-observed)<=1e-8,'MILP optimum differs')
  chosen_milp=np.rint(res.x.reshape(128,3));req(np.all(chosen_milp.sum(axis=1)==1) and float((chosen_milp*np.array([1,2,4])).sum())<=256,'MILP assignment feasibility')
  rep=report['tasks'][task];req(rep['questions']==128 and rep['selected_counts']=={str(k):counts[k] for k in (128,256,512)} and rep['total_visual_tokens']==used and rep['questions_with_different_k']==sum(k!=256 for k in chosen) and math.isclose(rep['oracle_mean_pct'],float(exact*100/128),abs_tol=1e-10) and math.isclose(rep['fixed_k256_mean_pct'],float(base*100/128),abs_tol=1e-10) and math.isclose(rep['oracle_minus_fixed_pp'],float((exact-base)*100/128),abs_tol=1e-10),'reported task')
  tasks[task]={'assigned_score_sum':str(exact),'fixed_k256_score_sum':str(base),'used_tokens':used,'selected_counts':{str(k):counts[k] for k in (128,256,512)},'milp_double_optimum':other,'milp_gap':float(res.mip_gap),'milp_dual_bound':float(-res.mip_dual_bound),'absolute_difference_vs_assignment':abs(other-observed)}
 out={'status':'passed','independent_review':True,'questions':384,'tasks':tasks,'analysis_sha256':sha(O/'analysis.json'),'assignments_sha256':sha(O/'assignments.jsonl'),'score_sha256':sha(source),'plan_sha256':sha(D/'plan001.json'),'analysis_started_sha256':sha(D/'monitor_analysis001/started.json'),'analysis_terminal_sha256':sha(D/'monitor_analysis001/terminal.json'),'reviewer_sha256':sha(__file__),'solver':'scipy.optimize.milp HiGHS','scipy_version':scipy.__version__,'GT_opened':False,'new_predictions_opened':False,'limitations':['MILP is double-precision numerical optimization; assignment objective agrees within 1e-8, not a proof of exact decimal tie-breaking.','Answer-informed in-sample upper bound on these DEV scores among three precomputed arms, not a deployable policy or generalization result.','Per-task sum of visual K does not measure TTFT, FLOPs, memory or guarantee equal compute.'],'errors':[]}
 with a.out.open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('independent MILP verification passed')
if __name__=='__main__':main()
