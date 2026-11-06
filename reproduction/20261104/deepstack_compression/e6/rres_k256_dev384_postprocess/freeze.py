"""Freeze the proposed one-shot orchestration, without starting it."""
import hashlib,json,os
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];F=B/'e6/rres_k256_dev384';E=B/'e6/rres_k256_dev384_exec';I=B/'e1_independent_review';A=B/'e6/rres_k256_full_analysis_v2';PY='/media/disk2/YZX/research/EADP_v2/_reproduction/envs/qwen_author/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def step(name,mon,worker,out,product,reserved,status,required,timeout):
 cmd=[PY,str(B/'monitor/supervise.py'),'--run-dir',str(mon),'--timeout',str(timeout),'--snapshot-seconds','3600','--poll-seconds','5','--grace-seconds','60','--',PY,str(worker)]
 if out is not None:cmd+=['--out',str(out)]
 return dict(name=name,monitor=str(mon),worker=str(worker),out_arg=str(out) if out else None,product=str(product),reserved=str(reserved),status=status,required=required,timeout=timeout,grace=60,command=cmd)
steps=[
 step('snapshot',F/'monitor_snapshot001',E/'snapshot.py',None,F/'source_snapshot001/manifest.json',F/'source_snapshot001','passed',{},1800),
 step('formal',F/'monitor_verify001',E/'verify.py',F/'verification001.json',F/'verification001.json',F/'verification001.json','passed',{'images':384,'GT_opened':False,'scoring_performed':False,'errors':[]},1800),
 step('independent_result',I/'monitor_rres_k256_dev384_result',I/'review_rres_k256_dev384_result.py',I/'rres_k256_dev384_result_review.json',I/'rres_k256_dev384_result_review.json',I/'rres_k256_dev384_result_review.json','passed',{'independent_review':True,'images':384,'GT_opened':False,'scoring_performed':False,'errors':[]},3600),
 step('score',A/'monitor_score001',A/'score.py',A/'score001',A/'score001/analysis.json',A/'score001','completed',{'questions':384},3600),
 step('independent_score',I/'monitor_rres_k256_dev384_scores',I/'review_rres_k256_dev384_scores.py',I/'rres_k256_dev384_scores_review.json',I/'rres_k256_dev384_scores_review.json',I/'rres_k256_dev384_scores_review.json','passed',{'independent_review':True,'questions':384,'GT_opened':True,'scoring_performed':True,'errors':[]},3600)]
sources=[D/'coordinator.py',Path(__file__),B/'initial_plan.json',B/'monitor/supervise.py',F/'gpu_plan001.json',F/'verification_plan001.json',I/'rres_k256_dev384_gpu_prelaunch_review.json',A/'score_plan001.json',I/'rres_k256_full_score_prelaunch_review.json']+[Path(x['worker']) for x in steps]
plan={'scope':'one-shot postprocessing after clean generation; no GPU launch','coordinator_sha256':sha(D/'coordinator.py'),'sources':{str(p):sha(p) for p in sources},'budget_end_unix':json.loads((B/'initial_plan.json').read_text())['budget_end_unix'],'steps':steps,'limitations':['canonical paths are exclusively assigned to this coordinator','pre-existing monitor or output stops; no automatic retry or takeover','scoring reads DEV truth only after scorer dual-acceptance gate','run snapshot, formal and reviewers retain their own checks','supervisor timeout values for postprocessing are newly fixed here rather than in the GPU plan']}
with (D/'plan001.json').open('x') as f:json.dump(plan,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(sha(D/'plan001.json'))
