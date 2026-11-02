"""Official DocVQA evaluator + independent ANLS, only after root actual exit."""
import importlib.util,json,os,sys
from pathlib import Path
import common as c

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 plan=c.read(c.PLAN);c.check(plan,'container')
 proof=c.read(c.HERE/'root_terminal_process_proof.json');release=c.read(c.HERE/'execution/release.json')
 from contracts import require_score_gate
 require_score_gate(proof,release,c.sha(c.PLAN),c.sha(c.HERE/'execution/release.json'),c.sha(c.HERE/'execution/generation_verification.json'))
 assert not (c.HERE/'failure.json').exists() and not (c.HERE/'worker_failure.json').exists()
 state=c.read(c.HERE/'state.json');claim=c.read(c.HERE/'launch.claim.json')
 allocation=c.HERE/'root_gpu_allocation.json';launch_path=c.HERE/'root_launch_process_snapshot.json'
 launch=c.read(launch_path);verification=c.read(c.HERE/'execution/generation_verification.json')
 complete=c.read(c.HERE/'execution/native_complete.json')
 assert state['status']=='complete' and state['release_sha256']==c.sha(c.HERE/'execution/release.json')
 assert state['native_exit_code']==state['CPU_verifier_exit_code']==0 and state['original_container_exited'] and state['GPU_pid_absent']
 assert proof['controller_state_sha256']==c.sha(c.HERE/'state.json') and proof['launch_snapshot_sha256']==c.sha(launch_path)
 assert proof['allocation_sha256']==c.sha(allocation)==state['allocation_sha256']==claim['allocation_sha256']==release['allocation_sha256']
 assert launch['root_exec_handle']==proof['root_owned_exec_handle'] and launch['plan_sha256']==c.sha(c.PLAN)
 assert launch['allocation_sha256']==proof['allocation_sha256'] and launch['controller']==state['controller']==claim['controller']
 assert launch['docker_client']==state['docker_client'] and launch['container_identity']==state['container_identity']
 assert proof['container_id']==state['container_identity']['container_id'] and proof['container_not_running']
 expected={'controller':state['controller'],'docker_client':state['docker_client'],'host_init':state['container_identity']['host_init']}
 checks={r['role']:r for r in proof['process_checks']};assert set(checks)==set(expected)
 for role,original in expected.items():assert checks[role]['original']==original and checks[role]['original_exited']
 assert verification['success'] and verification['n_questions']==5349 and verification['n_generations']==10698
 assert verification['plan_sha256']==c.sha(c.PLAN) and verification['native_complete_sha256']==c.sha(c.HERE/'execution/native_complete.json')
 assert verification['processor_identity_checks']==verification['independent_token_redecodes']==verification['prefill_intervention_checks']==10698 and verification['endpoint_anchors']==verification['fixed16_variant_anchors']==32
 assert verification['CPU_only'] and not verification['model_loaded'] and not verification['CUDA_initialized']
 assert verification['runtime_sha256']==c.sha(c.HERE/'execution/runtime.json')
 assert not verification['GT_read'] and not verification['scoring_started']
 assert complete['success'] and complete['n_questions']==5349 and complete['n_generations']==10698 and complete['endpoint_anchors']==32 and complete['plan_sha256']==c.sha(c.PLAN)
 assert complete['prediction_sha256']==verification['prediction_sha256'] and complete['observation_sha256']==verification['observation_sha256']
 endpoints=c.read(c.HERE/'execution/endpoints_verified.json')
 assert endpoints['success'] and endpoints['n_endpoints']==32 and endpoints['plan_sha256']==c.sha(c.PLAN)
 assert c.sha(c.HERE/'execution/endpoints_verified.json')==complete['endpoints_verified_sha256']==verification['endpoints_verified_sha256']
 for arm in ('plain','eadp1024'):
  assert c.sha(c.HERE/'execution'/('endpoint_'+arm+'.predictions.jsonl'))==endpoints['prediction_sha256'][arm]
  assert c.sha(c.HERE/'execution'/('endpoint_'+arm+'.observations.jsonl'))==endpoints['observation_sha256'][arm]
 assert complete['all_generation_before_scoring'] and not complete['GT_read'] and not complete['scoring_started']
 for arm in ('plain_ds_off','eadp1024_ds_on'):
  assert c.sha(c.HERE/'execution'/(arm+'.predictions.jsonl'))==complete['prediction_sha256'][arm]
  assert c.sha(c.HERE/'execution'/(arm+'.observations.jsonl'))==complete['observation_sha256'][arm]
 # Validate all rows and order against the answer-free input manifest before GT.
 manifest=c.rows(c.BASE/'input_manifest.jsonl');assert len(manifest)==5349
 predictions={arm:c.rows(c.HERE/'execution'/(arm+'.predictions.jsonl')) for arm in ('plain_ds_off','eadp1024_ds_on')}
 for arm in predictions:
  assert len(predictions[arm])==5349
  for item,pred in zip(manifest,predictions[arm]):
   assert pred['index']==item['index'] and pred['position']==item['position'] and pred['native_failure'] is None and isinstance(pred['prediction'],str)
 for arm in predictions:
  assert c.sha(c.HERE/'execution'/(arm+'.predictions.jsonl'))==complete['prediction_sha256'][arm]
  assert c.sha(c.HERE/'execution'/(arm+'.observations.jsonl'))==complete['observation_sha256'][arm]
 for path,digest in plan['score_only_sha256'].items():assert c.sha(path)==digest,path
 # Root actual exit gate is complete: this is the first GT parsing and score call in this scoring stage; preparation and hash checks did read canonical bytes.
 canonical=c.rows(c.Q/'independent_audit/DocVQA_VAL.canonical.jsonl')
 assert len(canonical)==len(manifest)==5349
 for pos,(g,m) in enumerate(zip(canonical,manifest)):
  assert str(g['index'])==m['index'] and g['question']==m['question'] and m['position']==pos
 out=c.HERE/'score';out.mkdir(exist_ok=False)
 import pandas as pd
 old=c.read(c.OLD/'run_plan.json');sys.path.insert(0,old['author_root']);sys.path.insert(0,str(Path(old['author_root'])/'VLMEvalKit'))
 from vlmeval.dataset.image_vqa import ImageVQADataset
 audit_path=c.Q/'independent_audit/audit_qwen.py';spec=importlib.util.spec_from_file_location('qualified_audit_qwen',audit_path)
 audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
 import torch
 assert not torch.cuda.is_initialized()
 dataset=ImageVQADataset.__new__(ImageVQADataset);dataset.dataset_name='DocVQA_VAL'
 scores={}
 for arm in ('plain_ds_off','eadp1024_ds_on'):
  raw=predictions[arm];table=pd.DataFrame(canonical);table['prediction']=[r['prediction'] for r in raw]
  xlsx=out/(arm+'.xlsx');table.to_excel(xlsx,index=False)
  # Native evaluator performs the same Excel reload, pandas NA conversion,
  # official process_line/hit_calculate and saves its detailed+aggregate files.
  aggregate=dataset.evaluate(str(xlsx));detail=out/(arm+'_results.xlsx');saved=pd.read_excel(detail)
  assert len(saved)==5349
  per=[]
  for gt,pred,native in zip(canonical,raw,saved.to_dict(orient='records')):
   assert audit.normalized_id(native['index'])==str(gt['index'])
   text=audit.scoring_prediction(pred['prediction']);truth=audit.parse_answers(gt['answer'])
   independent=audit.independent_anls(text,truth)
   native_distances=__import__('ast').literal_eval(str(native['eval_match']))
   official=1.-min(native_distances);official=official if official>=.5 else 0.
   assert abs(independent-official)<1e-12
   per.append(dict(index=str(gt['index']),prediction=pred['prediction'],scored_prediction=text,
     native_NA_conversion=pred['prediction']!=text,official_score=official,independent_score=independent))
  score=100*sum(r['independent_score'] for r in per)/5349
  assert 'Overall' in aggregate.columns and len(aggregate)==1
  official=float(aggregate['Overall'].iloc[0])
  assert abs(official-score)<1e-10
  target=out/(arm+'.per_question.jsonl')
  with target.open('x') as f:
   for row in per:f.write(json.dumps(row,ensure_ascii=False)+'\n')
  scores[arm]=dict(official=official,independent=score,all_pointwise_equal=True,
   NA_conversions=sum(r['native_NA_conversion'] for r in per),empty_answers=sum(r['prediction']=='' for r in per),
   per_question_sha256=c.sha(target),native_detail_sha256=c.sha(detail),native_input_sha256=c.sha(xlsx))
 pairs=[]
 for arm,baseline in (('plain_ds_off','plain'),('eadp1024_ds_on','eadp1024')):
  new=c.rows(out/(arm+'.per_question.jsonl')); oldrows=c.rows(c.BASE/'score'/(baseline+'.per_question.jsonl'))
  assert len(new)==len(oldrows)==5349
  for pos,(x,y) in enumerate(zip(new,oldrows)):
   assert x['index']==y['index'];pairs.append(dict(arm=arm,position=pos,index=x['index'],baseline=y['independent_score'],toggle=x['independent_score'],difference=x['independent_score']-y['independent_score'],answer_changed=x['prediction']!=y['prediction']))
 target=out/'paired_rows.jsonl'
 with target.open('x') as f:
  for row in pairs:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 assert not torch.cuda.is_initialized()
 c.save(out/'score_report.json',dict(success=True,n=5349,scores=scores,
  per_arm_comparison={arm:dict(score_difference=scores[arm]['independent']-c.read(c.BASE/'score/score_report.json')['scores'][baseline]['independent'],changed_answers=sum(r['answer_changed'] for r in pairs if r['arm']==arm),improved=sum(r['difference']>0 for r in pairs if r['arm']==arm),declined=sum(r['difference']<0 for r in pairs if r['arm']==arm)) for arm,baseline in (('plain_ds_off','plain'),('eadp1024_ds_on','eadp1024'))},paired_rows_sha256=c.sha(target),
  all_generation_and_root_exit_before_first_score=True,full_denominators_preserved=True,CPU_only=True,model_loaded=False,
  CUDA_initialized=False,plan_sha256=c.sha(c.PLAN),limits=plan['limits']))
 print(json.dumps(dict(success=True,n=5349,scores=scores)))
if __name__=='__main__':main()
