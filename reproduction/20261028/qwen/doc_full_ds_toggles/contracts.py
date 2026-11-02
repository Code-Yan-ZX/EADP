"""Pure CPU contracts for raw preservation and root terminal score gates."""
import json

def preserve_raw(stream,item,prediction,failure):
 row=dict(position=item['position'],index=item['index'],prediction=prediction,native_failure=failure)
 stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();return row

def check_observation(item,prediction,failure,obs):
 assert failure is None and isinstance(prediction,str) and not obs['observer_errors']
 assert obs['actual_processor']==item['processor_expected'] and obs['decode_matches'] and obs['num_image_tokens']==1024

def require_score_gate(proof,release,plan_sha,release_sha,verification_sha):
 assert proof['root_owned_exec_actual_exit_code']==0 and proof['all_original_processes_exited']
 assert proof['plan_sha256']==plan_sha and proof['release_sha256']==release_sha
 assert proof['generation_verification_sha256']==verification_sha
 assert release['success'] and release['native_container_exited'] and release['plan_sha256']==plan_sha
 assert release['generation_verification_sha256']==verification_sha
