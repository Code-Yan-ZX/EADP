"""Independent scipy recomputation, executable only after complete accepted analysis."""
import argparse,hashlib,importlib.util,json,math,os
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import rankdata,spearmanr
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');I=B/'e1_independent_review';D=B/'e4_multistream_expansion343_analysis'
TASKS=('DocVQA_VAL','TextVQA_VAL','AI2D_TEST');NAMES=('main_cv','ds0_cv','ds1_cv','ds2_cv','main_ds0_rank_corr','main_ds1_rank_corr','main_ds2_rank_corr','four_stream_rank_disagreement')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(d,script,out=None):
 s=read(d/'started.json');t=read(d/'terminal.json');c=s['command'];cwd=Path(s['cwd']);assert len(c)>=2 and (cwd/c[1]).resolve()==script.resolve() and s['worker_pid']==t['worker_pid'] and t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[]
 if out is not None:assert c.count('--out')==1 and (cwd/c[c.index('--out')+1]).resolve()==out.resolve()
def key(x):return x['task'],str(x['qid'])
def feature(n):
 assert n.shape==(4,1024) and n.dtype==np.float64 and np.isfinite(n).all() and (n>=0).all()
 ranks=[(rankdata(v,method='average')-1)/1023 for v in n];out={}
 for i in range(4):out[NAMES[i]]=None if n[i].mean()==0 else float(n[i].std(ddof=0)/n[i].mean())
 for i in range(3):out[NAMES[4+i]]=None if np.ptp(ranks[0])==0 or np.ptp(ranks[i+1])==0 else float(spearmanr(n[0],n[i+1]).statistic)
 out[NAMES[7]]=float(np.mean(np.var(np.asarray(ranks),axis=0,ddof=0)));return out
def close(a,b):return a is None and b is None or (a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(a,b,abs_tol=1e-12,rel_tol=1e-12))
def summary(rows):
 assert len(rows)==384 and len({key(r) for r in rows})==384;out={};groups={}
 for task in TASKS:
  subset=[x for x in rows if x['task']==task];groups[task]={x['group_id'] for x in subset};assert len(subset)==len(groups[task])==128
  results={}
  for name in NAMES:
   pairs=[(x['features'][name],x['scores'][2]-x['scores'][0]) for x in subset if x['features'][name] is not None];n=len(pairs);xx=[x for x,_ in pairs];yy=[y for _,y in pairs]
   why='fewer_than_three_defined_pairs' if n<3 else ('constant_feature' if len(set(xx))==1 else ('constant_outcome' if len(set(yy))==1 else None))
   results[name]={'rho':None if why else float(spearmanr(xx,yy).statistic),'reason':why,'n_task':128,'feature_defined':n,'feature_undefined':128-n,'outcome_defined':128,'pairs':n,'feature_tied_observations':n-len(set(xx)),'outcome_tied_observations':n-len(set(yy))}
  out[task]=results
 return {'questions':384,'per_task':128,'correlations':out,'cross_task_group_overlap':{x+'|'+y:len(groups[x]&groups[y]) for i,x in enumerate(TASKS) for y in TASKS[i+1:]}}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out).resolve();expected=I/'e4_multistream_expansion343_analysis_review.json';assert out==expected and not out.exists()
 analysis=D/'analysis002.json';terminal(D/'monitor_analysis002',D/'analyze.py',analysis)
 # Complete 343 formal+independent gate and frozen analysis proof precede every new record read.
 code=read(D/'code_plan003.json');gp=D/'gate.py';assert sha(gp)==code['source_sha256'][str(gp)]
 sp=importlib.util.spec_from_file_location('_independent_multistream_analysis_gate',gp);g=importlib.util.module_from_spec(sp);sp.loader.exec_module(g);evidence=g.accepted()
 result=read(analysis);assert result['status']=='passed' and result['questions']==384 and result['cohort_counts']==[32,9,343] and result['GT_opened'] is False and result['scoring_performed'] is False and result['predictions_opened'] is True and result['upstream']==evidence and result['errors']==[]
 orig=read(B/'e5/selector_real32/sample_manifest_001.json')['pilot'];s=read(B/'e4_multistream_expansion352/selection001.json');pilot=s['pilot'];pilot_keys={key(x) for x in pilot};rest=[x for x in s['remaining'] if key(x) not in pilot_keys];cohort=[*orig,*pilot,*rest];assert (len(orig),len(pilot),len(rest),len({key(x) for x in cohort}))==(32,9,343,384)
 old={key(x):x for x in read(B/'e4_multistream_budget_diagnostic_v2/result001.json')['rows']};featuremap={k:r['features'] for k,r in old.items()};assert len(featuremap)==32
 for selected,run in ((pilot,B/'e4_multistream_expansion352_gpu_v4/run001'),(rest,B/'e4_multistream_expansion343_gpu/run001')):
  for i,item in enumerate(selected):
   k=key(item);rec=read(run/'captures'/('image_%03d.json'%i));norm=np.load(run/'captures'/('image_%03d_norms.npy'%i),allow_pickle=False);computed=feature(norm)
   assert k not in featuremap and key(rec)==k and all(str(rec[t])==str(item[t]) for t in ('group_id','image_sha256','message_sha256')) and set(rec['features'])==set(computed) and all(close(rec['features'][t],computed[t]) for t in NAMES);featuremap[k]=computed
 scores={}
 with (B/'e4_analysis/score_panel003/scores.jsonl').open() as fp:
  for line in fp:
   v=json.loads(line);k=key(v);assert k not in scores
   vals=[]
   for n in (128,256,512):
    arm=v['arms'][f'K{n}-D1'];x=float(arm['score']);y=float(arm['independent_score']);assert math.isfinite(x) and 0<=x<=1 and abs(x-y)<=1e-12;vals.append(x)
   scores[k]=(v['group_id'],vals)
 assert len(scores)==384 and set(scores)==set(featuremap)=={key(x) for x in cohort};rmap={key(x):x for x in result['rows']};assert len(rmap)==384
 for target in cohort:
  k=key(target);x=rmap[k];assert all(str(x[t])==str(target[t]) for t in ('group_id','image_sha256','message_sha256')) and scores[k][0]==x['group_id'] and x['scores']==scores[k][1] and close(x['delta512_128'],x['scores'][2]-x['scores'][0]) and set(x['features'])==set(NAMES) and all(close(x['features'][t],featuremap[k][t]) for t in NAMES)
 independently=summary(list(rmap.values()));observed=result['summary'];assert independently['questions']==observed['questions'] and independently['per_task']==observed['per_task'] and independently['cross_task_group_overlap']==observed['cross_task_group_overlap']
 for task in TASKS:
  assert set(observed['correlations'][task])==set(NAMES)
  for name in NAMES:
   a1=independently['correlations'][task][name];a2=observed['correlations'][task][name];assert set(a1)==set(a2) and close(a1['rho'],a2['rho']) and all(a1[k]==a2[k] for k in a1 if k!='rho')
 proof={'status':'passed','independent_review':True,'questions':384,'task_correlations':24,'GT_opened':False,'scoring_performed':False,'predictions_opened':True,'analysis_sha256':sha(analysis),'analysis_started_sha256':sha(D/'monitor_analysis002/started.json'),'analysis_terminal_sha256':sha(D/'monitor_analysis002/terminal.json'),'formal_343_sha256':sha(B/'e4_multistream_expansion343_launch/verification001.json'),'review_343_sha256':sha(I/'e4_multistream_expansion343_result_review.json'),'reviewer_sha256':sha(__file__),'errors':[],'limitations':['Posthoc same-DEV associations; original 32 and pilot nine were previously inspected.','384 data include cross-task image overlap and are not held-out validation.','343 and pilot tensors were not retained; norm-to-model association is online capture evidence.']}
 with out.open('x') as f:json.dump(proof,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
