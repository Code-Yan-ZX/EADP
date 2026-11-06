"""Independent posthoc DEV geometry analysis arithmetic; run only after supervision gate."""
import argparse,hashlib,json,math,os
from pathlib import Path
B=Path('/media/disk2/YZX/research/EADP_v2/reproduction/20261104/deepstack_compression');D=B/'e4_adaptive_geometry_v2';R=B/'e1_independent_review'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open() as f:return [json.loads(x) for x in f if x.strip()]
def terminal(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);cmd=s['command'];assert t['reason']=='completed' and t['worker_returncode']==0 and t['remaining_process_group']==[] and s['worker_pid']==t['worker_pid'] and [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]==[script.resolve()]
 if out is not None:assert cmd.count('--out')==1 and (cwd/cmd[cmd.index('--out')+1]).resolve()==out.resolve()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
 mon=D/'monitor_analysis001';analysis=D/'analysis001';terminal(mon,D/'analyze.py',analysis)
 terminal(R/'monitor_e4_adaptive_geometry_v2_route_prelaunch',R/'review_e4_adaptive_geometry_v2_route_prelaunch.py',R/'e4_adaptive_geometry_v2_route_prelaunch_review.json')
 pre=read(R/'e4_adaptive_geometry_v2_route_prelaunch_review.json');assert pre['status']=='passed' and pre['analysis_freeze_sha256']==sha(D/'analysis_freeze005.json') and pre['reviewer_sha256']==sha(R/'review_e4_adaptive_geometry_v2_route_prelaunch.py')
 result=read(analysis/'analysis.json');assert result['status']=='completed' and result['GT_opened'] is False and result['new_predictions_opened'] is False and result['assignments_sha256']==sha(analysis/'assignments.jsonl')
 feats=lines(D/'features001/features.jsonl');scores=lines(B/'e4_analysis/score_panel003/scores.jsonl');actual=lines(analysis/'assignments.jsonl')
 key=lambda x:(x['task'],str(x['qid']),x['group_id']);F={key(x):x for x in feats};S={key(x):x for x in scores}
 assert len(F)==len(S)==384 and set(F)==set(S) and len(actual)==2304 and result['admission']['feature_sha256']==sha(D/'features001/features.jsonl') and result['admission']['scores_sha256']==sha(B/'e4_analysis/score_panel003/scores.jsonl')
 actual_by={(x['task'],str(x['qid']),x['group_id'],x['rule']):x for x in actual};assert len(actual_by)==2304
 h=lambda s:hashlib.sha256(s.encode()).hexdigest()
 for task in ('DocVQA_VAL','TextVQA_VAL','AI2D_TEST'):
  rows=[x for x in feats if x['task']==task];assert len(rows)==128 and len({x['group_id'] for x in rows})==128
  baseline=sum(S[key(x)]['arms']['K256-D1']['score'] for x in rows)/128
  observed=result['tasks'][task];assert observed['n']==128 and math.isclose(observed['fixed_k256_mean_pct'],100*baseline,abs_tol=1e-10)
  sorts={'nearest_low':lambda x:(x['nearest_mean'],h('20261105|'+x['group_id'])),'occupied8_high':lambda x:(-x['occupied8'],h('20261105|'+x['group_id'])),'entropy4_high':lambda x:(-x['entropy4'],h('20261105|'+x['group_id']))}
  for seed in range(3):sorts[f'random_{seed}']=lambda x,seed=seed:h(f"20261105|random|{seed}|"+x['group_id'])
  assert set(observed['rules'])==set(sorts)
  for name,sort in sorts.items():
   high={x['group_id'] for x in sorted(rows,key=sort)[:42]};values=[]
   for x in rows:
    k=512 if x['group_id'] in high else 128;record=S[key(x)]['arms'];score=record[f'K{k}-D1']['score'];ref=record['K256-D1']['score'];v=actual_by[(*key(x),name)]
    assert v['selected_k']==k and v['score']==score and v['baseline_score']==ref;values.append((score,ref))
   item=observed['rules'][name];assert item['high_k_count']==42 and item['mean_k']==254 and item['improved']==sum(x>y for x,y in values) and item['declined']==sum(x<y for x,y in values) and item['unchanged']==sum(x==y for x,y in values)
   assert math.isclose(item['mean_pct'],100*sum(x for x,_ in values)/128,abs_tol=1e-10) and math.isclose(item['minus_fixed256_pp'],100*(sum(x for x,_ in values)/128-baseline),abs_tol=1e-10)
 out={'status':'passed','independent_review':True,'scope':'posthoc DEV 384 feature/score paired routing arithmetic','questions':384,'rules_per_task':6,'assignments_verified':2304,'GT_opened':False,'existing_DEV_scores_opened':True,'analysis_sha256':sha(analysis/'analysis.json'),'assignments_sha256':sha(analysis/'assignments.jsonl'),'analysis_started_sha256':sha(mon/'started.json'),'analysis_terminal_sha256':sha(mon/'terminal.json'),'prelaunch_sha256':sha(R/'e4_adaptive_geometry_v2_route_prelaunch_review.json'),'features_sha256':sha(D/'features001/features.jsonl'),'scores_sha256':sha(B/'e4_analysis/score_panel003/scores.jsonl'),'reviewer_sha256':sha(__file__),'limitations':['Posthoc and batch-level; all six frozen rules and denominators retained.','Mean token count is not actual compute or latency; possible extra generation is not costed.'],'errors':[]}
 with open(args.out,'x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__':main()
