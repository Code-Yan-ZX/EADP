"""Freeze CPU candidate and historical source evidence. No GPU is launched."""
import sys
from pathlib import Path
import common as c

def main():
 assert sys.version_info[:2]==(3,10),'Author AST proof requires Python 3.10'
 assert not c.PLAN.exists() and not (c.HERE/'execution').exists() and not (c.HERE/'launch.claim.json').exists()
 import cpu_feasibility
 cpu_feasibility.main()
 old=c.read(c.OLD/'run_plan.json');factor=c.read(c.FACTOR/'execution/independent_verification.json');root=c.read(c.FACTOR/'root_terminal_process_proof.json')
 assert factor['success'] and factor['n_native_generations']==64 and root['root_owned_exec_actual_exit_code']==0
 assert root['independent_verification_sha256']==c.sha(c.FACTOR/'execution/independent_verification.json')
 refs=[c.HERE/n for n in ['README.md','common.py','contracts.py','intervention.py','selection_observer.py','historical_anchor.py','worker_draft.py','worker.py','prepare_plan.py','controller.py','cpu_feasibility.py','cpu_tests.py','cpu_worker_hook_test.py','cpu_old_anchor_test.py','cpu_container_entry.py','run_cpu_container_tests.py','cpu_preflight.py','cpu_container_check.py','verify_terminal.py','test_verify_terminal.py']]
 refs += [c.OLD/'run_plan.json',c.OLD/'root_terminal_process_proof.json',c.OLD/'execution/summary.json',c.OLD/'execution/per_question.jsonl',c.OLD/'independent_verification.json',c.FACTOR/'plan.json',c.FACTOR/'root_terminal_process_proof.json',c.FACTOR/'execution/independent_verification.json',c.AUDIT/'public_fa2_full_run02.terminal_audit.report.json',c.AUDIT/'public_fa2_full_run02_DocVQA_VAL.per_question.jsonl']
 source=[c.read(c.OLD/'execution/summary.json'),c.read(c.OLD/'root_terminal_process_proof.json')]
 assert source[1]['summary_sha256']==c.sha(c.OLD/'execution/summary.json') and source[0]['per_question_sha256']==c.sha(c.OLD/'execution/per_question.jsonl')
 with (c.OLD/'execution/per_question.jsonl').open() as f:records=[__import__('json').loads(line) for line in f]
 for i in range(16):
  f=c.OLD/'execution'/f'question_{i:02d}.npz';assert c.sha(f)==records[i]['cache']['sha256'];refs.append(f)
  r=c.FACTOR/'execution'/f'{i:02d}_eadp1024.json';a=c.FACTOR/'execution'/f'{i:02d}_eadp1024.npz';assert c.read(r)['archive_sha256']==c.sha(a);refs.extend((r,a))
 p=dict(run_id=c.RUN_ID,status='CPU_prepared_not_started',panel=old['panel'],model_instances=1,expected_generations=32,native_anchor_count=16,endpoint_first=True,files={str(x):c.sha(x) for x in refs},policy='All sixteen native K256 original responses, actual author selection and old cache/history anchors must pass before any DS-on call; pairs use one model and separately recompute, then bytewise compare actual selection and input tensors. Actual responses are saved before observation gates. No automatic queue/retry.',limits=['Unregistered DeepStack injection changes the published implementation; this diagnostic is not the reported paper score or a new pruning method.', 'Historical K256 generation saved response text but not output token IDs or prefill state.', 'Old fixed16 visual arrays are used only as sixteen source anchors, not as a new K256 DeepStack experiment.', 'No ground truth or scoring in this experiment; the previous completed historical prediction audit contains old score fields which are unused.', 'Same prefix continuation comparison cannot attribute post-divergence differences to a single cause.', 'Full outputs and high-dimensional tensors are retained; instrumentation affects timing.'])
 c.save(c.PLAN,p);c.check(p);print(c.sha(c.PLAN))
if __name__=='__main__':main()
