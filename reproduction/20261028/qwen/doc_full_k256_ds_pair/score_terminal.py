"""Separate official + independent scoring after complete root exit proof.

Never invoked by worker or controller. This module reads the answer-bearing
canonical file only after the gates below. It does not use old score values.
"""
import importlib.util,json,os,sys
from pathlib import Path
import common as c

def gate():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not any(Path('/dev',n).exists() for n in ('nvidia0','nvidiactl','nvidia-uvm'))
 p=c.read(c.PLAN);c.check(p,'container');plan_sha=c.sha(c.PLAN);out=c.HERE/'execution'
 assert not (c.HERE/'worker_failure.json').exists() and not (c.HERE/'failure.json').exists()
 proof=c.read(c.HERE/'root_terminal_process_proof.json');state=c.read(c.HERE/'state.json');claim=c.read(c.HERE/'launch.claim.json')
 release=c.read(out/'release.json');verification=c.read(out/'independent_verification.json');complete=c.read(out/'native_complete.json')
 allocation=c.HERE/'root_gpu_allocation.json';launch_path=c.HERE/'root_launch_process_snapshot.json';launch=c.read(launch_path)
 assert proof['root_owned_exec_actual_exit_code']==0 and proof['all_original_processes_exited'] and proof['container_not_running']
 assert state['status']=='complete' and state['native_exit_code']==state['CPU_verifier_exit_code']==0 and state['original_container_exited'] and state['GPU_pid_absent']
 assert proof['plan_sha256']==state['plan_sha256']==claim['plan_sha256']==release['plan_sha256']==verification['plan_sha256']==complete['plan_sha256']==plan_sha
 assert proof['controller_state_sha256']==c.sha(c.HERE/'state.json') and proof['launch_snapshot_sha256']==c.sha(launch_path)
 assert proof['allocation_sha256']==c.sha(allocation)==state['allocation_sha256']==claim['allocation_sha256']==release['allocation_sha256']
 assert launch['root_exec_handle']==proof['root_owned_exec_handle'] and launch['plan_sha256']==plan_sha and launch['allocation_sha256']==proof['allocation_sha256']
 assert launch['controller']==state['controller']==claim['controller'] and launch['docker_client']==state['docker_client'] and launch['container_identity']==state['container_identity']
 assert proof['container_id']==state['container_identity']['container_id']
 roles={'controller':state['controller'],'docker_client':state['docker_client'],'host_init':state['container_identity']['host_init']}
 checked={x['role']:x for x in proof['process_checks']};assert set(checked)==set(roles)
 for role,identity in roles.items():assert checked[role]['original']==identity and checked[role]['original_exited']
 assert release['success'] and release['native_container_exited'] and state['release_sha256']==proof['release_sha256']==c.sha(out/'release.json')
 assert release['independent_verification_sha256']==proof['independent_verification_sha256']==c.sha(out/'independent_verification.json')
 assert verification['success'] and verification['n_questions']==5349 and verification['n_generations']==10698 and verification['native_anchors']==32
 assert verification['processor_identity_checks']==verification['selection_checks']==verification['prefill_intervention_checks']==verification['independent_token_redecodes']==10698
 assert verification['CPU_only'] and not verification['model_loaded'] and not verification['CUDA_initialized'] and not verification['GT_read'] and not verification['scoring_started']
 assert verification['native_complete_sha256']==c.sha(out/'native_complete.json') and verification['runtime_sha256']==c.sha(out/'runtime.json')
 assert complete['success'] and complete['n_questions']==5349 and complete['n_generations']==10698 and complete['native_anchors']==32 and not complete['GT_read'] and not complete['scoring_started']
 for key in ('prediction_sha256','observation_sha256','ordered_prediction_sha256','ordered_observation_sha256'):
  assert verification[key]==complete[key]
 for arm in ('ds_off','ds_on'):
  for key,name in [('prediction_sha256',arm+'.predictions.jsonl'),('observation_sha256',arm+'.observations.jsonl'),('ordered_prediction_sha256',arm+'.ordered.predictions.jsonl'),('ordered_observation_sha256',arm+'.ordered.observations.jsonl')]:
   assert c.sha(out/name)==complete[key][arm]
 assert complete['native_anchors_sha256']==verification['native_anchors_sha256']==c.sha(out/'native_anchors.json')
 anchors=c.read(out/'native_anchors.json');assert anchors['success'] and anchors['n']==32 and anchors['plan_sha256']==plan_sha
 for key in anchors['records']:
  assert c.sha(out/'anchors'/(key+'.json'))==anchors['records'][key] and c.sha(out/'anchors'/(key+'.npz'))==anchors['archives'][key]
 items=c.rows(c.INPUT);predictions={a:c.rows(out/(a+'.ordered.predictions.jsonl')) for a in ('ds_off','ds_on')}
 for arm,rows in predictions.items():
  assert len(rows)==len(items)==5349
  for item,row in zip(items,rows):assert row['position']==item['position'] and row['index']==item['index'] and row['arm']==arm and row['native_failure'] is None and isinstance(row['response'],str)
 return p,items,predictions

def main():
 p,items,predictions=gate()
 # This is the first GT-byte access in this stage. Hashes were copied from the
 # already frozen previous plan, without reading GT during this preparation.
 for path,digest in p['score_only_sha256'].items():assert c.sha(path)==digest,path
 canonical_path=c.Q/'independent_audit/DocVQA_VAL.canonical.jsonl';canonical=c.rows(canonical_path)
 assert len(canonical)==len(items)==5349
 for pos,(gt,item) in enumerate(zip(canonical,items)):assert str(gt['index'])==item['index'] and gt['question']==item['question'] and item['position']==pos
 target=c.HERE/'score';target.mkdir(exist_ok=False)
 import pandas as pd
 old=c.read(c.OLD/'run_plan.json');sys.path.insert(0,old['author_root']);sys.path.insert(0,str(Path(old['author_root'])/'VLMEvalKit'))
 from vlmeval.dataset.image_vqa import ImageVQADataset
 audit_path=c.Q/'independent_audit/audit_qwen.py';spec=importlib.util.spec_from_file_location('qualified_audit_qwen',audit_path);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
 import torch
 assert not torch.cuda.is_initialized();dataset=ImageVQADataset.__new__(ImageVQADataset);dataset.dataset_name='DocVQA_VAL';scores={}
 for arm in ('ds_off','ds_on'):
  raw=predictions[arm];table=pd.DataFrame(canonical);table['prediction']=[r['response'] for r in raw];xlsx=target/(arm+'.xlsx');table.to_excel(xlsx,index=False)
  aggregate=dataset.evaluate(str(xlsx));detail=target/(arm+'_results.xlsx');saved=pd.read_excel(detail);assert len(saved)==5349;per=[]
  for gt,pred,native in zip(canonical,raw,saved.to_dict(orient='records')):
   assert audit.normalized_id(native['index'])==str(gt['index']);text=audit.scoring_prediction(pred['response']);truth=audit.parse_answers(gt['answer'])
   independent=audit.independent_anls(text,truth);distances=__import__('ast').literal_eval(str(native['eval_match']));official=1.-min(distances);official=official if official>=.5 else 0.
   assert abs(independent-official)<1e-12;per.append(dict(index=str(gt['index']),prediction=pred['response'],scored_prediction=text,native_NA_conversion=pred['response']!=text,official_score=official,independent_score=independent))
  independent=100*sum(r['independent_score'] for r in per)/5349;assert 'Overall' in aggregate.columns and len(aggregate)==1
  official=float(aggregate['Overall'].iloc[0]);assert abs(official-independent)<1e-10
  path=target/(arm+'.per_question.jsonl')
  with path.open('x') as f:
   for row in per:f.write(json.dumps(row,ensure_ascii=False)+'\n')
  scores[arm]=dict(official=official,independent=independent,all_pointwise_equal=True,NA_conversions=sum(r['native_NA_conversion'] for r in per),per_question_sha256=c.sha(path),native_detail_sha256=c.sha(detail),native_input_sha256=c.sha(xlsx))
 a=c.rows(target/'ds_off.per_question.jsonl');b=c.rows(target/'ds_on.per_question.jsonl');pairs=[]
 for position,(x,y) in enumerate(zip(a,b)):
  assert x['index']==y['index'];pairs.append(dict(position=position,index=x['index'],off=x['independent_score'],on=y['independent_score'],difference=y['independent_score']-x['independent_score'],answer_changed=x['prediction']!=y['prediction']))
 paired=target/'paired_rows.jsonl'
 with paired.open('x') as f:
  for row in pairs:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 assert not torch.cuda.is_initialized()
 c.save(target/'score_report.json',dict(success=True,n=5349,scores=scores,ds_on_minus_off=scores['ds_on']['independent']-scores['ds_off']['independent'],changed_answers=sum(x['answer_changed'] for x in pairs),improved=sum(x['difference']>0 for x in pairs),declined=sum(x['difference']<0 for x in pairs),paired_rows_sha256=c.sha(paired),all_generation_and_root_exit_before_first_score=True,full_denominators_preserved=True,CPU_only=True,model_loaded=False,CUDA_initialized=False,plan_sha256=c.sha(c.PLAN),limits=p['limits']))
 print(json.dumps(dict(success=True,n=5349,scores=scores)))
if __name__=='__main__':main()
