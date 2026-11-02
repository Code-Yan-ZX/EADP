"""Freeze unlabeled full K256 pair candidate; no score or GPU access."""
import sys
import common as c

def main():
 assert sys.version_info[:2]==(3,10),'The author AST proof requires Python 3.10'
 assert not c.PLAN.exists() and not (c.HERE/'execution').exists() and not (c.HERE/'launch.claim.json').exists()
 old=c.read(c.OLD/'run_plan.json');previous=c.read(c.FIXED/'plan.json');panel=previous['panel']
 assert panel==old['panel'] and len(panel)==16
 root=c.read(c.FIXED/'root_exit_proof.json');verify=c.read(c.FIXED/'execution/independent_verification.json');complete=c.read(c.FIXED/'execution/native_complete.json')
 assert root['root_pty_actual_exit_code']==0 and root['original_container_stopped'] and all(x['original_process_absent'] for x in root['roles'])
 assert root['independent_verification_sha256']==c.sha(c.FIXED/'execution/independent_verification.json') and root['release_sha256']==c.sha(c.FIXED/'execution/release.json')
 assert verify['success'] and verify['n_generations']==32 and verify['native_anchors']==16 and verify['native_complete_sha256']==c.sha(c.FIXED/'execution/native_complete.json')
 assert verify['runtime_sha256']==c.sha(c.FIXED/'execution/runtime.json') and complete['success']
 names=['README.md','common.py','contracts.py','intervention.py','selection_observer.py','worker_draft.py','worker.py','controller.py','prepare_plan.py','cpu_tests.py','cpu_worker_hook_test.py','cpu_reorder_test.py','cpu_score_gate_test.py','cpu_worker_archive_test.py','reorder.py','cpu_container_entry.py','run_cpu_container_tests.py','cpu_preflight.py','cpu_container_check.py','score_terminal.py','verify_generation.py','test_verify_generation.py']
 score_plan=c.Q/'doc_full_two_routes/plan.json';score_only=c.read(score_plan)['score_only_sha256']
 refs=[c.HERE/n for n in names]+[score_plan,c.INPUT,c.OLD/'run_plan.json',c.FIXED/'plan.json',c.FIXED/'root_exit_proof.json',c.FIXED/'execution/independent_verification.json',c.FIXED/'execution/native_complete.json',c.FIXED/'execution/native_anchors.json',c.FIXED/'execution/runtime.json',c.FIXED/'execution/release.json']
 for i in range(16):
  for arm in ('ds_off','ds_on'):
   key=f'{i:02d}_{arm}';record=c.FIXED/'execution'/(key+'.json');archive=c.FIXED/'execution'/(key+'.npz');raw=c.FIXED/'execution'/(key+'.raw.json')
   assert c.sha(record)==complete['records'][key] and c.sha(archive)==complete['archives'][key]
   assert c.read(record)['archive_sha256']==c.sha(archive) and c.read(record)['raw_sha256']==c.sha(raw)
   refs.extend((record,archive,raw))
 p=dict(run_id=c.RUN_ID,status='CPU_prepared_not_started',rows=5349,model_instances=1,panel=panel,expected_generations=10698,native_anchor_count=32,files={str(x):c.sha(x) for x in refs},score_only_sha256=score_only,policy='All 16 native off anchors and 16 on anchors byte-close before any remaining question, then one model and two separately selected native calls per remaining question. Responses are preserved before inspection. No retry or automatic GPU queue.',limits=['Same legacy-position paired implementation diagnosis, not the paper score or a new pruning method.','Old full K256 internal states were not saved; 16 previous exact paired arrays are the historical internal anchor.','For other 5333 pairs, actual high-dimensional arrays are hashed but not saved; independent verification cannot reconstruct those tensor values.','The old input preparation had previously read canonical data to construct the answer-free manifest; this task does not read GT or score files.','Changing DS can naturally change KV and generated suffix; compare continuation only for common generated prefixes.','Runtime and disk cost are instrumentation-dependent and not a model performance benchmark.'])
 c.save(c.PLAN,p);c.check(p);print(c.sha(c.PLAN))
if __name__=='__main__':main()
