"""Freeze CPU-only candidate plus 32 accepted raw endpoint artifacts."""
from pathlib import Path
import common as c

def main():
 assert not c.PLAN.exists() and not (c.HERE/'execution').exists()
 previous=c.read(c.PREV/'plan.json');report=c.read(c.PREV/'execution/independent_verification.json')
 root=c.read(c.PREV/'root_terminal_process_proof.json');release=c.read(c.PREV/'execution/release.json')
 score=c.read(c.FULL/'score_terminal_process_proof.json')
 assert report['success'] and report['n_native_generations']==32 and report['n_questions']==16
 assert root['root_owned_exec_actual_exit_code']==0 and root['all_original_processes_exited']
 assert root['independent_verification_sha256']==c.sha(c.PREV/'execution/independent_verification.json')
 assert root['release_sha256']==c.sha(c.PREV/'execution/release.json') and release['success']
 assert score['root_owned_score_exec_actual_exit_code']==0 and score['scored_questions_per_arm']==5349
 assert score['score_report_sha256']==c.sha(c.FULL/'score/score_report.json')
 refs=[c.HERE/n for n in ('common.py','intervention.py','worker_draft.py','prepare_plan.py','cpu_tests.py','README.md',
  'cpu_torch_hook_test.py','cpu_torch_author_preflight.py','cpu_container_check.py','cpu_preflight.py','controller.py',
  'verify_terminal.py','test_verify_terminal.py','run_verifier_tests.py')]
 refs += [c.PREV/'plan.json',c.PREV/'root_terminal_process_proof.json',c.PREV/'execution/independent_verification.json',
          c.PREV/'execution/release.json',c.FULL/'score_terminal_process_proof.json',c.FULL/'score/score_report.json']
 for i in range(16):
  for arm in ('plain','eadp1024'):
   rec_path=c.PREV/'execution'/f'{i:02d}_{arm}.json';rec=c.read(rec_path)
   path=c.PREV/'execution'/f'{i:02d}_{arm}.npz';assert Path(rec['archive'])==path and c.sha(path)==rec['archive_sha256']
   refs.extend((rec_path,path))
 p=dict(run_id=c.RUN_ID,status='CPU_prepared_draft_no_launcher',panel=previous['panel'],
  files={str(path):c.sha(path) for path in refs},
  arms=['plain','eadp1024','plain_ds_off','eadp1024_ds_on'],endpoint_first=True,
  expected_generations=64,model_instances=1,
  policy='Repeat and preserve all32 native endpoints; require bitwise visual/prefill/position/cache/rope/outputs anchors before 32 prefill-only DeepStack toggles.',
  limits=['The native pathways also differ in input API and potentially continuation state; this is not a fully isolated position-factor experiment.',
   'Actual DeepStack injection is unmeasured until native execution; no GT, scoring, or answer selection.',
   'Worker requires independent code review, CPU container preflight, synthetic author-runtime hook test, verifier tests, and explicit root allocation.'])
 c.save(c.PLAN,p);c.check(p)
 c.save(c.HERE/'READY.json',dict(status='CPU_draft_only_not_GPU_ready',plan_sha256=c.sha(c.PLAN),GPU_started=False,model_loaded=False))
 print(c.sha(c.PLAN))
if __name__=='__main__':main()
