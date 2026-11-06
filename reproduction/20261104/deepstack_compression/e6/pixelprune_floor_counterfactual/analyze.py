"""Post-hoc output routing using already-scored DEV records; no model inference."""
import argparse,hashlib,importlib.util,json,math
from pathlib import Path
B=Path(__file__).resolve().parent.parent.parent;D=B/'e6/pixelprune_floor_counterfactual';P=B/'e6/pixelprune_t045_dev384_analysis/paired.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def key(x):return (x['task'],str(x['qid']),x['group_id'])
def unique(rows):
 out={key(x):x for x in rows}
 if len(out)!=len(rows):raise ValueError('duplicate identity')
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();plan=read(D/'plan001.json')
 src45,src0=B/plan['t045_scores'],B/plan['t0_scores'];rev45,rev0=read(B/plan['t045_review']),read(B/plan['t0_review'])
 for proof,src in ((rev45,src45),(rev0,src0)):
  if proof.get('status')!='passed' or proof.get('independent_review') is not True or proof.get('questions')!=384 or proof.get('score_sha256')!=sha(src):raise ValueError('unverified scored records')
 x45,x0=unique(lines(src45)),unique(lines(src0))
 if len(x45)!=384 or set(x45)!=set(x0):raise ValueError('paired inventory')
 spec=importlib.util.spec_from_file_location('fixed_pixel_paired',P);paired=importlib.util.module_from_spec(spec);spec.loader.exec_module(paired)
 if Path(paired.__file__).resolve()!=P:raise ValueError('paired source identity')
 arms=('routed','pixel_t045','pixel_t0','eadp_k256_d1');contrasts={'routed_minus_t045':['routed','pixel_t045'],'routed_minus_t0':['routed','pixel_t0'],'routed_minus_eadp_k256':['routed','eadp_k256_d1']}
 reports={};details=[]
 for task in plan['tasks']:
  taskkeys=sorted((k for k in x45 if k[0]==task),key=lambda k:k[1].encode())
  if len(taskkeys)!=128 or len({k[2] for k in taskkeys})!=128:raise ValueError('task group inventory')
  reports[task]={}
  for route in plan['routes']:
   observations=[];budgets=[];fallbacks=0
   for k in taskkeys:
    rec,old=x45[k],x0[k];k45=rec['selected_count'];k0=old['selected_count']
    if type(k45) is not int or type(k0) is not int or not(1<=k45<=1024 and 1<=k0<=1024):raise ValueError('invalid K')
    a45,a0,ref=rec['arms']['pixel_t045'],rec['arms']['pixel_t0'],rec['arms']['e4_k256_d1'];old0=old['arms']['pixel_dynamic']
    if a0['answer']!=old0['answer'] or a0['eos']!=old0['eos'] or abs(a0['score']-old0['score'])>1e-12:raise ValueError('t0 identity')
    fallback=k45<=route['use_t0_when_t045_k_le'];chosen=a0 if fallback else a45;budget=k0 if fallback else k45;fallbacks+=fallback;budgets.append(budget)
    values={'routed':chosen,'pixel_t045':a45,'pixel_t0':a0,'eadp_k256_d1':ref}
    scores={name:v['score'] for name,v in values.items()};answers={name:v.get('answer',v.get('raw_answer')) for name,v in values.items()}
    if any(not math.isfinite(v) or not 0<=v<=1 for v in scores.values()):raise ValueError('invalid score')
    observations.append({'group_id':k[2],'scores':scores,'answers':answers})
    details.append({'task':k[0],'qid':k[1],'group_id':k[2],'route':route['name'],'k045':k45,'k0':k0,'fallback_to_t0':fallback,'selected_k':budget,'scores':scores})
   result=paired.analyze(observations,list(arms),contrasts,replicates=10000,seed=20261104)
   b=sorted(budgets);result['dynamic_k']={'min':b[0],'mean':sum(b)/128,'median':(b[63]+b[64])/2,'p95_nearest_rank':b[math.ceil(.95*128)-1],'max':b[-1],'fallbacks':fallbacks}
   reports[task][route['name']]=result
 a.out.mkdir(exist_ok=False)
 with (a.out/'routes.jsonl').open('x') as f:
  for x in details:f.write(json.dumps(x,sort_keys=True)+'\n')
 result={'status':'completed','plan_sha256':sha(D/'plan001.json'),'sources':{'t045_scores':sha(src45),'t0_scores':sha(src0),'t045_review':sha(B/plan['t045_review']),'t0_review':sha(B/plan['t0_review']),'paired_source':sha(P)},'routes_sha256':sha(a.out/'routes.jsonl'),'tasks':reports,'GT_opened':False,'new_predictions_opened':False,'limitations':plan['limitations']}
 with (a.out/'analysis.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print('completed')
if __name__=='__main__':main()
