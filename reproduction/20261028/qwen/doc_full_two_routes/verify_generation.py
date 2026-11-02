"""Post-generation CPU container audit; no GT payloads or scoring."""
import json,os,sys
from pathlib import Path
import common as c

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 p=c.read(c.PLAN);c.check(p,'container')
 state=c.read(c.HERE/'state.json');assert state['status']=='native_exited_validation_pending' and state['native_exit_code']==0
 assert state['original_container_exited'] and state['GPU_pid_absent']
 out=c.HERE/'execution';runtime=c.read(out/'runtime.json');assert runtime['post_process'] is False and runtime['plan_sha256']==c.sha(c.PLAN)
 done=c.read(out/'native_complete.json')
 assert done['success'] and done['counts']=={'plain':5349,'eadp1024':5349} and done['plan_sha256']==c.sha(c.PLAN)
 assert done['all_generation_before_scoring'] and not done['GT_read'] and not done['scoring_started']
 items=c.rows(c.HERE/'input_manifest.jsonl');predictions={};observations={}
 for arm in ('plain','eadp1024'):
  pred=out/(arm+'.predictions.jsonl');obs=out/(arm+'.observations.jsonl')
  assert c.sha(pred)==done['predictions'][arm] and c.sha(obs)==done['observations'][arm]
  predictions[arm]=c.rows(pred);observations[arm]=c.rows(obs)
  assert len(predictions[arm])==len(observations[arm])==len(items)==5349
  for i,(item,r,o) in enumerate(zip(items,predictions[arm],observations[arm])):
   assert r['position']==o['position']==item['position']==i and r['index']==o['index']==item['index']
   assert r['native_failure'] is None and isinstance(r['prediction'],str)
   assert o['native_failure'] is None and not o['observer_errors']
   assert o['actual_processor']==item['processor_expected'] and o['num_image_tokens']==1024 and o['decode_matches']
   assert o['native_redecode']==r['prediction'] and len(o['output_suffix_ids'])<=2048
   assert all(type(x) is int and 0<=x for x in o['output_suffix_ids'])
 import torch
 from transformers import AutoProcessor
 assert not torch.cuda.is_initialized()
 old=c.read(c.OLD/'run_plan.json');tokenizer=AutoProcessor.from_pretrained(old['recipe']['model_path'],local_files_only=True).tokenizer
 for arm in ('plain','eadp1024'):
  decoded=tokenizer.batch_decode([o['output_suffix_ids'] for o in observations[arm]],skip_special_tokens=True,clean_up_tokenization_spaces=False)
  assert decoded==[r['prediction'] for r in predictions[arm]]
 assert not torch.cuda.is_initialized()
 for path,digest in p['file_guards'].items():assert c.sha(path)==digest,path
 target=out/'generation_verification.json';assert not target.exists()
 c.save(target,dict(success=True,n_questions=5349,n_generations=10698,plan_sha256=c.sha(c.PLAN),
  native_complete_sha256=c.sha(out/'native_complete.json'),prediction_sha256=done['predictions'],observation_sha256=done['observations'],
  processor_identity_checks=10698,independent_token_redecodes=10698,native_container_exit_verified_by_controller=True,
  CPU_only=True,model_loaded=False,CUDA_initialized=False,GT_read=False,scoring_started=False,
  root_controller_actual_exit_proof_required=True))
 print(json.dumps(dict(success=True,n_questions=5349,n_generations=10698)))
if __name__=='__main__':main()
