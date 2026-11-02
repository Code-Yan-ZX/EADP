"""Freeze reviewed CPU-only implementation and prior accepted evidence; no GPU launch."""
from pathlib import Path
import sys
import common as c

def main():
 assert sys.version_info[:2]==(3,10), 'The locked AST proof requires the author Python 3.10 interpreter.'
 assert not c.PLAN.exists() and not (c.HERE/'execution').exists() and not (c.HERE/'launch.claim.json').exists()
 factor=c.read(c.PANEL/'execution/independent_verification.json'); fr=c.read(c.PANEL/'root_terminal_process_proof.json')
 base=c.read(c.BASE/'score/score_report.json'); br=c.read(c.BASE/'root_terminal_process_proof.json'); sr=c.read(c.BASE/'score_terminal_process_proof.json')
 assert factor['success'] and factor['n_native_generations']==64 and factor['endpoint_anchors']==32
 assert fr['root_owned_exec_actual_exit_code']==br['root_owned_exec_actual_exit_code']==0
 assert fr['all_original_processes_exited'] and br['all_original_processes_exited'] and base['success']
 assert sr['root_owned_score_exec_actual_exit_code']==0 and sr['scorer_container_not_running'] and sr['score_report_sha256']==c.sha(c.BASE/'score/score_report.json')
 assert c.read(c.BASE/'score_cpu_smoke_result.json')['actual_exit']==0
 assert sr['generation_root_proof_sha256']==c.sha(c.BASE/'root_terminal_process_proof.json') and sr['scored_questions_per_arm']==5349
 for arm in ('plain','eadp1024'):
  assert c.sha(c.BASE/'score'/f'{arm}.per_question.jsonl')==base['scores'][arm]['per_question_sha256']
 files=['common.py','contracts.py','prepare_plan.py','worker.py','controller.py','cpu_container_check.py','cpu_preflight.py','verify_generation.py','test_verify_generation.py','score_terminal.py','cpu_tests.py','cpu_torch_hook_test.py','run_cpu_torch_hook_test.py','README.md']
 refs=[c.HERE/x for x in files]+[c.OLD/'run_plan.json',c.PANEL/'plan.json',c.PANEL/'root_terminal_process_proof.json',c.PANEL/'execution/independent_verification.json',c.BASE/'input_manifest.jsonl',c.BASE/'plan.json',c.BASE/'root_terminal_process_proof.json',c.BASE/'score_terminal_process_proof.json',c.BASE/'score/score_report.json',c.BASE/'execution/generation_verification.json',c.BASE/'execution/native_complete.json',c.BASE/'execution/release.json',c.BASE/'score_cpu_smoke_result.json',c.BASE/'score_cpu_smoke.py',c.BASE/'run_score_cpu_smoke.py']
 for a in ('plain','eadp1024'):
  refs += [c.BASE/'execution'/f'{a}.{suffix}.jsonl' for suffix in ('predictions','observations')]+[c.BASE/'score'/f'{a}.per_question.jsonl']
 for i in range(16):
  for a in ('plain','eadp1024','plain_ds_off','eadp1024_ds_on'):
   record=c.PANEL/'execution'/f'{i:02d}_{a}.json'; archive=c.PANEL/'execution'/f'{i:02d}_{a}.npz'
   assert c.read(record)['archive_sha256']==c.sha(archive)
   refs.extend([record,archive])
 refs.append(c.PANEL/'intervention.py')
 old=c.read(c.OLD/'run_plan.json');src=Path(old['author_root'])/'VLMEvalKit/vlmeval/dataset'
 score_paths=[c.Q/'independent_audit/DocVQA_VAL.canonical.jsonl',c.Q/'independent_audit/audit_qwen.py',src/'image_vqa.py',src/'utils/vqa_eval.py']
 plan=dict(run_id=c.RUN_ID,status='prepared_not_started',rows=5349,model_instances=1,file_guards={str(x):c.sha(x) for x in refs},score_only_sha256={str(x):c.sha(x) for x in score_paths},limits=['Diagnostic of implementation paths, not a paper benchmark or a new method.', 'Old full run lacks per-question internal tensor records and complete runtime numerical flags; only 16 endpoint and variant anchors are tested.', 'Actual tensor metadata is captured, but high-dimensional tensors are not retained; recorded digests cannot be independently reconstructed.', 'Interventions alter subsequent KV naturally; they do not directly edit cache.', 'Prior preparation read canonical answers to construct the answer-free manifest; this generation does not read ground truth.', 'No scoring before full generation, independent CPU validation, and actual root/controller exit proof.'])
 c.save(c.PLAN,plan);c.check(plan);print(c.sha(c.PLAN))
if __name__=='__main__':main()
