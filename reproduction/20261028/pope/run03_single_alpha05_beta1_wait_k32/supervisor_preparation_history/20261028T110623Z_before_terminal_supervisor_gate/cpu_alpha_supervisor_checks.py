#!/usr/bin/env python3
"""CPU-only negative gate tests; never arms or launches a supervisor/model."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import supervise_alpha_after_next as supervisor
import verify_next_predecessor as predecessor
from verify_complete_predictions import verify


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    outcomes = []

    def rejected(name, function):
        try:
            function()
            raise RuntimeError('Invalid gate accepted: ' + name)
        except (AssertionError, RuntimeError) as error:
            if str(error).startswith('Invalid gate accepted:'):
                raise
            outcomes.append({'check': name, 'passed': True, 'expected_rejection': type(error).__name__})

    supervisor.frozen_gate()
    outcomes.append({'check': 'all_frozen_control_NeXT_Qwen_guards_pass', 'passed': True})
    ready, gate = predecessor.prerequisites()
    assert not ready and not supervisor.CLAIM.exists()
    outcomes.append({'check': 'live_partial_predecessor_does_not_open_gate', 'passed': True, 'reason': gate['reason']})
    owner = supervisor.root_slot_A_worker(2364896)
    assert owner['task'] == 'DocVQA_VAL'
    outcomes.append({'check': 'real_current_root_A_full_container_proc_argv_identity', 'passed': True})
    plan = supervisor.root_slot_A_plan()
    future = owner['cmdline'][:]
    future[future.index('--data') + 1] = 'AI2D_TEST'
    assert supervisor.validate_root_A_command(owner['cgroup'], future, plan) == 'AI2D_TEST'
    outcomes.append({'check': 'pure_predicate_future_native_AI2D_same_container_allowed', 'passed': True,
        'synthetic_argv_only_not_a_live_worker': True})
    rejected('foreign_container_cgroup_rejected', lambda: supervisor.validate_root_A_command('0::/docker/foreign\n', owner['cmdline'], plan))
    wrong = owner['cmdline'][:]
    wrong[wrong.index('--model') + 1] = 'unregistered-model'
    rejected('same_container_wrong_model_rejected', lambda: supervisor.validate_root_A_command(owner['cgroup'], wrong, plan))
    wrong_task = owner['cmdline'][:]
    wrong_task[wrong_task.index('--data') + 1] = 'MMBench_DEV_EN'
    rejected('same_container_unregistered_task_rejected', lambda: supervisor.validate_root_A_command(owner['cgroup'], wrong_task, plan))
    original_gpu = predecessor.gpu_apps
    try:
        predecessor.gpu_apps = lambda: ('synthetic two GPU PIDs', [{'pid': 1}, {'pid': 2}])
        assert not supervisor.capacity_gate()[0]
        outcomes.append({'check': 'synthetic_two_models_capacity_rejected', 'passed': True})
        predecessor.gpu_apps = lambda: ('synthetic registeredA-only snapshot', [{'pid': 2364896}])
        assert supervisor.capacity_gate()[0]
        outcomes.append({'check': 'registered_A_only_capacity_predicate_passes', 'passed': True,
            'synthetic_GPU_snapshot_not_current_global_ready': True})
        predecessor.gpu_apps = lambda: ('synthetic foreign CPU-PID-as-GPU snapshot', [{'pid': os.getpid()}])
        assert not supervisor.capacity_gate()[0]
        outcomes.append({'check': 'unknown_single_GPU_pid_not_whitelisted', 'passed': True})
    finally:
        predecessor.gpu_apps = original_gpu
    with tempfile.TemporaryDirectory(prefix='supervisor_gate_tests_', dir=supervisor.RUN) as tmp:
        temporary = Path(tmp)
        claim = temporary / 'exclusive.json'
        supervisor.exclusive_json(claim, {'test': True})
        try:
            supervisor.exclusive_json(claim, {'test': False})
            raise RuntimeError('duplicate exclusive launch accepted')
        except FileExistsError:
            outcomes.append({'check': 'one_attempt_exclusive_claim_rejects_duplicate', 'passed': True})
        original_run, original_next = predecessor.RUN, predecessor.NEXT
        try:
            predecessor.NEXT, predecessor.RUN = temporary, temporary / 'run'
            predecessor.RUN.mkdir()
            (predecessor.RUN / 'failure.json').write_text('{}')
            rejected('predecessor_failure_sidecar_stops_gate', predecessor.prerequisites)
            (predecessor.RUN / 'failure.json').unlink()
            score_path = predecessor.RUN / 'official.score.json'
            score_path.write_text(json.dumps({'n': 4999, 'token_decoding_checks': 4999}))
            (predecessor.RUN / 'state.json').write_text(json.dumps({'status': 'complete', 'n': 5000, 'pid': predecessor.PID}))
            (predecessor.NEXT / 'full_default_K128.supervisor_state.json').write_text(json.dumps({'status': 'complete', 'exit_code': 0}))
            rejected('synthetic_4999_terminal_score_rejected', predecessor.prerequisites)
        finally:
            predecessor.RUN, predecessor.NEXT = original_run, original_next
    evaluator_path = ROOT.parents[2] / '_reproduction/author_eadp/LLaVA/llava/eval/m4c_evaluator.py'
    spec = importlib.util.spec_from_file_location('author_consensus_boundary_test', evaluator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    official = module.TextVQAAccuracyEvaluator()
    for count in range(11):
        gt = ['yes'] * count + ['other'] * (10 - count)
        direct = official.eval_pred_list([{'pred_answer': 'yes', 'gt_answers': gt}])
        assert abs(direct - predecessor.consensus_from_count(count)) < 1e-12
    outcomes.append({'check': 'independent_GT_match_count_formula_matches_all11_consensus_boundaries', 'passed': True})
    nested = supervisor.RUN / 'supervisor_cpu_reference/nonexistent_nested_output'
    positive = verify(ROOT / 'run02_source_wait_only_k32/predictions.jsonl', nested)
    assert (nested / 'official_score.log').exists() and (nested / 'full_prediction_verification.json').exists()
    assert positive['n_predictions'] == 8910 and positive['score_log_sha256'] == supervisor.control.sha(nested / 'official_score.log')
    outcomes.append({'check': 'real_full8910_verifier_mkdir_nested_score_log_path', 'passed': True})
    supervisor.frozen_gate()
    assert 'torch' not in sys.modules and not supervisor.CLAIM.exists()
    assert not (supervisor.RUN / 'one_launch_attempt.json').exists()
    assert not (supervisor.RUN / 'process_identity.json').exists()
    for name in ['verify_next_predecessor.py', 'supervise_alpha_after_next.py']:
        compile((ROOT / name).read_text(), str(ROOT / name), 'exec')
    outcomes.append({'check': 'no_torch_no_arm_no_worker_no_attempt_and_frozen_bf40_unchanged', 'passed': True})
    report = {'all_passed': True, 'checks': outcomes, 'supervisor_started': False, 'GPU_model_loaded': False,
        'frozen_run03_plan_sha256': supervisor.control.sha(supervisor.RUN / 'run_plan.json'),
        'supervisor_preparation_sha256': supervisor.control.sha(supervisor.PREPARATION),
        'scope': 'CPU gate predicates and temporary labeled negative fixtures. Complete8910 reference only for nested scorer-output path; no current partial benchmark scoring.'}
    supervisor.control.save(supervisor.RUN / 'supervisor_cpu_checks.json', report)
    print(json.dumps({'checks_passed': len(outcomes), 'all_passed': True, 'supervisor_started': False,
        'GPU_model_loaded': False, 'frozen_plan_sha256': report['frozen_run03_plan_sha256']}))


if __name__ == '__main__':
    main()
