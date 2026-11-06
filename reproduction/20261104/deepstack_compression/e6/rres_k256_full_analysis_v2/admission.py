"""Fail closed before opening target predictions, prior scored rows or truth."""
import hashlib,importlib.metadata,json
from pathlib import Path
D=Path(__file__).resolve().parent;B=D.parents[1];R=B/'e1_independent_review';RUN=B/'e6/rres_k256_dev384';EXEC=B/'e6/rres_k256_dev384_exec';RAND=B/'e6/random_full_exec_v2';SCORE=B/'e6/random_full_analysis_v3'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def terminal(mon,script,out=None):
 s=read(mon/'started.json');t=read(mon/'terminal.json');cwd=Path(s['cwd']);cmd=s['command']
 if (type(s.get('worker_pid')) is not int or s['worker_pid']<=0 or s['worker_pid']!=t.get('worker_pid') or t.get('reason')!='completed' or t.get('worker_returncode')!=0 or t.get('remaining_process_group')!=[] or [(cwd/x).resolve() for x in cmd if Path(x).name==script.name]!=[script.resolve()]):raise ValueError('supervisor exit or command')
 if out is not None and (cmd.count('--out')!=1 or (cwd/cmd[cmd.index('--out')+1]).resolve()!=out.resolve()):raise ValueError('supervisor output')
def target():
 formal=RUN/'verification001.json';proof=R/'rres_k256_dev384_result_review.json';reviewer=R/'review_rres_k256_dev384_result.py';verifier=EXEC/'verify.py'
 terminal(RUN/'monitor_run001',EXEC/'launch.py');terminal(RUN/'monitor_snapshot001',EXEC/'snapshot.py')
 terminal(RUN/'monitor_verify001',verifier,formal);terminal(R/'monitor_rres_k256_dev384_result',reviewer,proof)
 f=read(formal);q=read(proof)
 fields={'formal_sha256':formal,'formal_started_sha256':RUN/'monitor_verify001/started.json','formal_terminal_sha256':RUN/'monitor_verify001/terminal.json','run_started_sha256':RUN/'monitor_run001/started.json','run_terminal_sha256':RUN/'monitor_run001/terminal.json','reviewer_sha256':reviewer}
 if (f.get('status')!='passed' or f.get('images')!=384 or f.get('GT_opened') is not False or f.get('scoring_performed') is not False or f.get('errors')!=[] or f.get('verifier_sha256')!=sha(verifier)
     or q.get('status')!='passed' or q.get('independent_review') is not True or q.get('images')!=384 or q.get('GT_opened') is not False or q.get('scoring_performed') is not False or q.get('errors')!=[] or any(q.get(k)!=sha(v) for k,v in fields.items())):raise ValueError('target dual acceptance')
 if not isinstance(f.get('files_sha256'),dict) or q.get('files_sha256')!=f['files_sha256']:raise ValueError('target file lists differ')
 if f.get('source_manifest_sha256')!=q.get('source_manifest_sha256') or f.get('source_manifest_sha256')!=sha(RUN/'source_snapshot001/manifest.json'):raise ValueError('target source snapshot binding')
 if any(sha(k)!=v for k,v in f['files_sha256'].items()):raise ValueError('target accepted file changed')
 anchors=read(B/'e6/pixelprune_t020_dev384/plan001.json')['anchors'];files=f['files_sha256'];directory=RUN/'run001/observations'
 if len(anchors)!=384 or len({(x['task'],str(x['qid']),x['group_id']) for x in anchors})!=384:raise ValueError('anchor inventory')
 for x in anchors:
  name=x['task']+'_'+str(x['qid'])
  if any(str((directory/(name+suffix)).resolve()) not in files for suffix in ('.json','_generated.json')):raise ValueError('target file inventory')
 return {'formal_sha256':sha(formal),'independent_sha256':sha(proof),'source_manifest_sha256':f['source_manifest_sha256']}
def references():
 e1=B/'e1/e1full001_completion.json';ei=R/'e1full001_review.json';ev=B/'e1_analysis/completion.py';er=R/'review_full.py';em=B/'e1/monitorverifye1full001';eim=R/'monitor_e1full001'
 terminal(em,ev,e1);terminal(eim,er,ei);p=read(e1);q=read(ei);journal=B/'e1/e1full001/raw_predictions.jsonl'
 if (p.get('status')!='passed' or p.get('natural_questions')!=384 or p.get('natural_records')!=1536 or p.get('GT_opened') is not False or p.get('scoring_performed') is not False or p.get('verifier_sha256')!=sha(ev) or p['files']['journal']['path']!=str(journal) or p['files']['journal']['sha256']!=sha(journal) or q.get('status')!='passed' or q.get('n_verified')!=384 or q.get('errors')!=[] or q.get('completion_proof_sha256')!=sha(e1) or q.get('completion_started_sha256')!=sha(em/'started.json') or q.get('completion_terminal_sha256')!=sha(em/'terminal.json') or q.get('reviewer_sha256')!=sha(er)):raise ValueError('E1 inference')
 erp=R/'e1full001_score_review.json';terminal(B/'e1/monitorscoree1full001',B/'e1_analysis/score.py',B/'e1/e1full001_score');terminal(R/'monitor_e1score001',R/'review_score.py');e=read(erp)
 frozen=read(B/'e1/e1full001_score/frozen.json')
 if any(sha(k)!=v for k,v in frozen['code_sha256'].items()) or any(importlib.metadata.version(k)!=v for k,v in frozen['runtime_versions'].items()) or e.get('status')!='passed' or e.get('n_verified')!=384 or e.get('arms_verified')!=1536 or e.get('errors')!=[] or e.get('score_sha256')!=sha(B/'e1/e1full001_score/scores.jsonl') or e.get('score_frozen_sha256')!=sha(B/'e1/e1full001_score/frozen.json') or e.get('reviewer_sha256')!=sha(R/'review_score.py') or e.get('e1_review_sha256')!=sha(ei):raise ValueError('E1 scoring')
 randomproof=R/'random_full_v2_scores_review.json';randomreview=R/'review_random_full_v2_scores.py';terminal(R/'monitor_random_full_v2_scores',randomreview,randomproof);terminal(SCORE/'monitor_score001',SCORE/'score.py',SCORE/'score001');rr=read(randomproof)
 rf=RAND/'verification001.json';ri=R/'random_full_v2_result_review.json';rrv=R/'review_random_full_v2_result.py';terminal(RAND/'monitor_run001',RAND/'launch.py');terminal(RAND/'monitor_snapshot001',RAND/'snapshot.py');terminal(RAND/'monitor_verify001',RAND/'verify.py',rf);terminal(R/'monitor_random_full_v2_result',rrv,ri);f=read(rf);ir=read(ri)
 identity={'formal_sha256':rf,'formal_started_sha256':RAND/'monitor_verify001/started.json','formal_terminal_sha256':RAND/'monitor_verify001/terminal.json','run_started_sha256':RAND/'monitor_run001/started.json','run_terminal_sha256':RAND/'monitor_run001/terminal.json','reviewer_sha256':rrv}
 if (f.get('status')!='passed' or f.get('questions')!=384 or f.get('generations')!=1152 or f.get('GT_opened') is not False or f.get('scoring_performed') is not False or f.get('errors')!=[] or f.get('verifier_sha256')!=sha(RAND/'verify.py') or f.get('runtime_plan_sha256')!=sha(RAND/'runtime_plan001.json') or ir.get('status')!='passed' or ir.get('independent_review') is not True or ir.get('GT_opened') is not False or ir.get('scoring_performed') is not False or any(ir.get(k)!=sha(v) for k,v in identity.items()) or ir.get('questions')!=384 or ir.get('generations')!=1152 or ir.get('errors')!=[]):raise ValueError('random inference')
 if (rr.get('status')!='passed' or rr.get('independent_review') is not True or rr.get('questions')!=384 or rr.get('random_generations')!=1152 or rr.get('errors')!=[] or rr.get('score_sha256')!=sha(SCORE/'score001/scores.jsonl') or rr.get('analysis_sha256')!=sha(SCORE/'score001/analysis.json') or rr.get('score_started_sha256')!=sha(SCORE/'monitor_score001/started.json') or rr.get('score_terminal_sha256')!=sha(SCORE/'monitor_score001/terminal.json') or rr.get('score_plan_sha256')!=sha(SCORE/'score_plan001.json') or rr.get('random_result_review_sha256')!=sha(ri) or rr.get('reviewer_sha256')!=sha(randomreview)):raise ValueError('random scores')
 records=str(RAND/'run_001/records.jsonl')
 if records not in f['files_sha256'] or ir.get('files_sha256',{}).get(records)!=f['files_sha256'][records] or sha(records)!=f['files_sha256'][records]:raise ValueError('random actual records')
 return {'e1_inference_sha256':sha(e1),'e1_review_sha256':sha(ei),'e1_score_review_sha256':sha(erp),'random_result_review_sha256':sha(ri),'random_score_review_sha256':sha(randomproof)}
def require():
 plan=read(D/'score_plan001.json')
 if plan.get('status')!='frozen' or plan.get('target_predictions_opened') is not False or plan.get('GT_opened') is not False or any(sha(k)!=v for k,v in plan['source_sha256'].items()):raise ValueError('source plan')
 terminal(D/'monitor_cpu001',D/'cpu_check.py');pre=R/'rres_k256_full_score_prelaunch_review.json';reviewer=R/'review_rres_k256_full_score_prelaunch.py';terminal(R/'monitor_rres_k256_full_score_prelaunch',reviewer,pre)
 q=read(pre);cpu=read(D/'cpu_proof001.json')
 if (q.get('status')!='passed' or q.get('independent_review') is not True or q.get('scope')!='rres_k256_full_score_prelaunch' or q.get('GT_opened') is not False or q.get('target_predictions_opened') is not False or q.get('errors')!=[] or q.get('score_plan_sha256')!=sha(D/'score_plan001.json') or q.get('cpu_proof_sha256')!=sha(D/'cpu_proof001.json') or q.get('reviewer_sha256')!=sha(reviewer) or cpu.get('status')!='passed' or cpu.get('source_sha256_after')!=plan['source_sha256']):raise ValueError('scoring prelaunch')
 target_evidence=target();reference_evidence=references()
 return {'source_sha256':plan['source_sha256'],'plan_sha256':sha(D/'score_plan001.json'),'target':target_evidence,'references':reference_evidence}
