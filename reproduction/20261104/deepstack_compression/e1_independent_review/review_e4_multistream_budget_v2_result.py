"""Independent complete DEV multi-stream score diagnostic recalculation; no GT."""
import argparse,hashlib,json,math,os
from collections import Counter
from pathlib import Path
import numpy as np
import torch
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');D=B/'e4_multistream_budget_diagnostic_v2';I=B/'e1_independent_review';S=B/'e5/selector_real32';E=B/'e4_panel_v2';X=B/'e4_analysis'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(1048576),b''):h.update(v)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def terminal(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');c=s['command'];cwd=Path(s['cwd']);assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in c if Path(x).name==script.name]==[script.resolve()]
 if out is not None:assert c.count('--out')==1 and (cwd/c[c.index('--out')+1]).resolve()==out.resolve()
def rank(x):
 # Independent average-tie ranks using distinct-value positions.
 v=np.asarray(x,dtype=np.float64);unique,inverse,counts=np.unique(v,return_inverse=True,return_counts=True);ends=np.cumsum(counts);starts=ends-counts+1;return ((starts+ends)/2)[inverse]
def corr(a,b):
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);a=a-np.mean(a);b=b-np.mean(b);den=np.sqrt(np.sum(a*a)*np.sum(b*b));return None if den==0 else float(np.sum(a*b)/den)
def close(a,b):
 if a is None or b is None:assert a is None and b is None
 else:assert math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();result=D/'result001.json';review=I/'e4_multistream_budget_v2_prelaunch_review.json';reviewer=I/'review_e4_multistream_budget_v2_prelaunch.py';terminal(D/'monitor_extract001',D/'extract.py',result);terminal(I/'monitor_e4_multistream_budget_v2_prelaunch',reviewer,review)
 f=read(D/'plan_freeze001.json');code=read(D/'code_plan001.json');pre=read(review);out=read(result);assert pre['status']=='passed' and pre['independent_review'] is True and pre['feature_or_score_values_opened'] is False and pre['reviewer_sha256']==sha(reviewer) and pre['plan_freeze_sha256']==sha(D/'plan_freeze001.json') and pre['code_plan_sha256']==sha(D/'code_plan001.json') and all(sha(k)==v for k,v in code['source_sha256'].items())
 assert out['status']=='passed' and out['questions']==32 and out['GT_opened'] is False and out['GPU_used'] is False and out['scoring_performed'] is False and out['predictions_opened'] is True and out['errors']==[] and out['source_sha256']==sha(D/'extract.py') and out['plan_freeze_sha256']==sha(D/'plan_freeze001.json') and out['code_plan_sha256']==sha(D/'code_plan001.json')
 manifest=S/'sample_manifest_001.json';sf=read(S/'run_001_verification.json');sr=read(I/'selector_real32_run001_review.json');pilot=read(manifest)['pilot'];assert sf['source_sha256'][str(manifest)]==sha(manifest) and sf['array_sha256']==sr['array_sha256'] and len(pilot)==32
 scorepath=X/'score_panel003/scores.jsonl';panelpath=E/'panel_003/records.jsonl';scoreproof=read(X/'independent_score_review/verification001.json');assert out['upstream']['scores_sha256']==scoreproof['score_sha256']==sha(scorepath) and out['upstream']['records_sha256']==read(E/'panel_003_verification.json')['files_sha256'][str(panelpath)]==sha(panelpath)
 targets={(x['task'],str(x['qid'])):x for x in pilot};assert len(targets)==32 and Counter(x['task'] for x in pilot)=={'DocVQA_VAL':11,'TextVQA_VAL':11,'AI2D_TEST':10}
 scores={}
 for x in lines(scorepath):
  key=(x['task'],str(x['qid']))
  if key not in targets:continue
  assert key not in scores and x['group_id']==targets[key]['group_id'];values=[]
  for k in (128,256,512):
   q=x['arms'][f'K{k}-D1'];v=float(q['score']);u=float(q['independent_score']);assert math.isfinite(v) and 0<=v<=1 and abs(v-u)<=1e-12;values.append(v)
  scores[key]=values
 assert len(scores)==32
 panels=set()
 for x in lines(panelpath):
  key=(x['task'],str(x['qid']));k=x['budget']
  if key not in targets or x['ds_on'] is not True or k not in (128,256,512):continue
  assert (key,k) not in panels and all(x[z]==targets[key][z] for z in ('task','group_id','message_sha256','image_sha256')) and x['actual_injection_verified'] is True;panels.add((key,k))
 assert len(panels)==96
 names=('main_cv','ds0_cv','ds1_cv','ds2_cv','main_ds0_rank_corr','main_ds1_rank_corr','main_ds2_rank_corr','four_stream_rank_disagreement');computed=[]
 for i,item in enumerate(pilot):
  path=S/'run_001'/('image_%03d'%i)/'capture.pt';assert sha(path)==f['capture_inventory_sha256'][str(path)]==sf['array_sha256'][str(path)]
  archive=torch.load(path,map_location='cpu',weights_only=True);main,ds=archive['actual']['visual'];assert len(ds)==3
  norms=[]
  for x in [main,*ds]:
   x=x.detach().to(dtype=torch.float64,device='cpu').numpy()
   if x.ndim==3 and x.shape[0]==1:x=x[0]
   assert x.ndim==2 and x.shape[0]==1024 and x.shape[1]>=1 and np.isfinite(x).all();norms.append(np.sqrt(np.sum(x*x,axis=1)))
  ranks=[(rank(x)-1)/1023 for x in norms];features={}
  for j,x in enumerate(norms):
   mean=float(np.mean(x));features[names[j]]=None if mean==0 else float(np.sqrt(np.mean((x-mean)**2))/mean)
  for j in range(3):features[names[4+j]]=corr(ranks[0],ranks[j+1])
  z=np.stack(ranks);center=np.mean(z,axis=0);features[names[7]]=float(np.mean(np.mean((z-center)**2,axis=0)))
  key=(item['task'],str(item['qid']));v=scores[key];row=out['rows'][i];assert all(str(row[k])==str(item[k]) for k in ('task','qid','group_id')) and set(row['features'])==set(names)
  for k in names:close(features[k],row['features'][k])
  for k,value in zip(('score128','score256','score512'),v):close(row[k],value)
  close(row['delta512_128'],v[2]-v[0]);computed.append((item,features,v[2]-v[0]));del archive,main,ds,x,norms,ranks,z
 for task,n in [('DocVQA_VAL',11),('TextVQA_VAL',11),('AI2D_TEST',10)]:
  group=[x for x in computed if x[0]['task']==task];assert len(group)==n and set(out['correlations'][task])==set(names)
  for name in names:
   pairs=[(feat[name],delta) for _,feat,delta in group if feat[name] is not None];r=out['correlations'][task][name]
   if len(pairs)<3:value=None;reason='fewer_than_three_defined_pairs'
   else:value=corr(rank([v[0] for v in pairs]),rank([v[1] for v in pairs]));reason='constant_feature_or_outcome' if value is None else None
   assert r['n_pairs']==len(pairs) and r['n_task']==n and r['reason']==reason;close(r['rho'],value)
 proof={'status':'passed','independent_review':True,'questions':32,'features_per_question':8,'task_feature_correlations':24,'GT_opened':False,'predictions_opened':True,'GPU_used':False,'scoring_performed':False,'result_sha256':sha(result),'result_started_sha256':sha(D/'monitor_extract001/started.json'),'result_terminal_sha256':sha(D/'monitor_extract001/terminal.json'),'prelaunch_sha256':sha(review),'reviewer_sha256':sha(__file__),'plan_sha256':sha(D/'plan_freeze001.json'),'scope':'independent per-capture features, accepted score identity and descriptive correlation recomputation','limitations':['32 images after earlier exploration; n=10/11 per task and no confirmatory inference.','Only existing accepted DEV scores/predictions accessed; no GT, training or new model generation.','Prior accepted E4 visual streams represented by hashes, not independently reconstructed full tensors.'],'errors':[]}
 with open(a.out,'x') as fp:json.dump(proof,fp,indent=2);fp.write('\n');fp.flush();os.fsync(fp.fileno())
if __name__=='__main__':main()
