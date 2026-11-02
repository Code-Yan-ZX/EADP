#!/usr/bin/env python3
"""Prepare/check a CPU waiter; one GPU run requires root's explicit slot B flag."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import alpha_control as control
import verify_next_predecessor as predecessor

ROOT, RUN = control.ROOT, control.RUN
PYTHON = '/home/dell/miniconda3/envs/llava_pruner/bin/python'
EXPECTED_CONTROL_PLAN = 'bf40e511c27cf920dbd287d61126c405346f97fc37728cff3254af608aef5a70'
SLOT_A = {'pid': 2364896, 'startticks': 1331402066}
PREPARATION = RUN / 'supervisor_preparation.json'
CLAIM = RUN / 'supervisor_launch_claim.json'
PURPOSE = ('One fixed single-alpha POPE control, alpha1->.5, beta1/K32/canonical8910/greedy128/same3wait unchanged. '
           'No score direction or paper-closeness threshold, retry, parameter search, or slot-A takeover.')


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(name, obj):
    control.save(RUN / name, obj)


def prepare():
    assert not CLAIM.exists() and not (RUN / 'process_identity.json').exists()
    plan = read(RUN / 'run_plan.json')
    assert control.sha(RUN / 'run_plan.json') == EXPECTED_CONTROL_PLAN
    control.validate_frozen(plan)
    predecessor.check_frozen_next()
    frozen_files = {str(path): control.sha(path) for path in [
        Path(__file__).resolve(), ROOT / 'verify_next_predecessor.py', RUN / 'run_plan.json',
        predecessor.PLAN, predecessor.WORKER]}
    record = {'status': 'prepared_not_started', 'prepared_utc': utc(), 'purpose': PURPOSE,
        'requires_future_explicit_root_slot_B_allocation': True,
        'assigned_slot_on_authorization': 'B', 'predecessor_worker': {'pid': predecessor.PID, 'startticks': predecessor.TICKS},
        'predecessor_plan_sha256': predecessor.EXPECTED_PLAN, 'predecessor_worker_sha256': predecessor.EXPECTED_WORKER,
        'control_plan_sha256': EXPECTED_CONTROL_PLAN, 'known_slot_A_worker': SLOT_A,
        'frozen_files': frozen_files, 'predecessor_independent_verification': str(predecessor.OUTPUT),
        'terminal_policy': 'Any predecessor/control failure is preserved and stops this supervisor; one launch claim, no automatic restart/retry.',
        'command_after_root_allocation': [PYTHON, str(Path(__file__).resolve()), 'run',
            '--root-allocated-slot-B', '--allocation-note', 'explicit root slot allocation after NeXTfull'],
        'GPU_loaded_or_supervisor_started': False}
    save(PREPARATION.name, record)
    return record


def frozen_gate():
    record = read(PREPARATION)
    assert record['status'] == 'prepared_not_started'
    assert control.sha(RUN / 'run_plan.json') == EXPECTED_CONTROL_PLAN
    for path, expected in record['frozen_files'].items():
        assert control.sha(path) == expected, 'Frozen supervisor/input changed: ' + path
    control.validate_frozen(read(RUN / 'run_plan.json'))
    predecessor.check_frozen_next()
    assert 'torch' not in sys.modules, 'Waiting supervisor must not import/load torch'
    return record


def capacity_gate():
    apps, processes = predecessor.gpu_apps()
    pids = {row['pid'] for row in processes}
    if predecessor.PID in pids:
        return False, {'reason': 'predecessor_still_present_on_GPU', 'nvidia_smi': apps}
    if len(pids) >= 2:
        return False, {'reason': 'waiting_global_GPU_process_count_below2', 'nvidia_smi': apps}
    unknown = pids - {SLOT_A['pid']}
    if unknown:
        return False, {'reason': 'unassigned_GPU_process_needs_root_review', 'unexpected_PIDs': sorted(unknown), 'nvidia_smi': apps}
    if SLOT_A['pid'] in pids:
        assert predecessor.identity(SLOT_A['pid']) == SLOT_A, 'Slot A PID identity changed'
    return True, {'reason': 'known_slot_A_or_empty_only', 'nvidia_smi': apps, 'GPU_process_count_before_loading': len(pids)}


def validate_independent_predecessor():
    result = read(predecessor.OUTPUT)
    assert result['success'] and result['n_rows'] == result['token_decoding_checks'] == result['per_question_score_checks'] == 5000
    assert result['plan_sha256'] == predecessor.EXPECTED_PLAN and result['worker_sha256'] == predecessor.EXPECTED_WORKER
    assert result['all_frozen_source_input_weight_image_guards_verified']
    assert abs(result['official_accuracy_percent'] - result['independent_accuracy_percent']) < 1e-10
    assert result['official_score_sha256'] == predecessor.sha(predecessor.RUN / 'official.score.json')
    for artifact in result['artifacts'].values():
        assert predecessor.sha(artifact['path']) == artifact['sha256']
    exited = result['worker_exit']
    assert (exited['pid'], exited['startticks']) == (predecessor.PID, predecessor.TICKS)
    assert exited['original_worker_exited'] and exited['gpu_pid_absent']
    ready, _ = predecessor.prerequisites()
    assert ready
    return result


def exclusive_json(path, obj):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(json.dumps(obj, indent=2) + '\n')


def complete_release():
    plan = read(RUN / 'run_plan.json')
    control.validate_frozen(plan)
    state = read(RUN / 'state.json')
    process = read(RUN / 'process_identity.json')
    summary_path = RUN / 'full_result_summary.json'
    summary = read(summary_path)
    assert state['status'] == 'complete' and state['generation_exit_code'] == state['score_exit_code'] == 0
    assert state['n_predictions'] == summary['n_predictions'] == 8910
    assert summary['complete'] and summary['gpu_worker_exit_verified']
    assert summary['frozen_source_inputs_payload_stats_verified'] and summary['actual_runtime_verified']
    assert summary['run_plan_sha256'] == process['plan_sha256'] == state['plan_sha256'] == EXPECTED_CONTROL_PLAN
    assert all(summary[key] for key in ['all_question_ids_order_verified', 'all_raw_prompts_verified', 'all_pinned_image_question_positions_verified'])
    worker = {'pid': process['worker_pid'], 'startticks': process['worker_start_ticks']}
    live = predecessor.identity(worker['pid'])
    assert live != worker, 'Actual POPE worker still alive'
    apps, processes = predecessor.gpu_apps()
    assert not any(row['pid'] == worker['pid'] for row in processes), 'POPE worker still present on GPU'
    independent = control.verify_predictions(RUN / 'predictions.jsonl', RUN / 'supervisor_prediction_verification')
    assert independent['predictions_sha256'] == summary['predictions_sha256']
    assert abs(independent['official_macro_f1_percent'] - summary['official_macro_f1_percent']) < 1e-12
    assert abs(summary['independent_macro_f1_percent'] - summary['official_macro_f1_percent']) < 1e-12
    assert control.sha(RUN / 'official_score.log') == summary['score_log_sha256']
    release = {'status': 'all_completion_and_exit_gates_passed', 'observed_utc': utc(),
        'GPU_slot': 'B', 'next_owner': '/root/v2_review', 'control_run': RUN.name,
        'worker_pid': worker['pid'], 'worker_start_ticks': worker['startticks'],
        'same_worker_still_alive': False, 'current_identity_if_pid_reused': live,
        'gpu_pid_absent': True, 'nvidia_smi': apps,
        'generation_exit_code': 0, 'official_score_exit_code': 0, 'prediction_count': 8910,
        'predictions_sha256': summary['predictions_sha256'], 'independent_and_official_macro_match': True,
        'all_source_sha_match': True, 'all_frozen_source_input_weight_image_guards_verified': True,
        'full_result_summary': {'path': str(summary_path), 'sha256': control.sha(summary_path)},
        'state_sha256': control.sha(RUN / 'state.json'), 'process_identity_sha256': control.sha(RUN / 'process_identity.json'),
        'run_plan_sha256': EXPECTED_CONTROL_PLAN, 'official_score_log_sha256': control.sha(RUN / 'official_score.log'),
        'independent_prediction_verification': {'path': str(RUN / 'supervisor_prediction_verification/full_prediction_verification.json'),
            'sha256': control.sha(RUN / 'supervisor_prediction_verification/full_prediction_verification.json')},
        'no_more_POPE_models_scheduled': True, 'score_direction_used_for_handoff': False}
    save('completion_and_release_proof.json', release)
    return release


def run(allocation_note):
    prepared = frozen_gate()
    assert not (RUN / 'predictions.jsonl').exists() and not (RUN / 'process_identity.json').exists()
    exclusive_json(CLAIM, {'claimed_utc': utc(), 'supervisor': predecessor.identity(os.getpid()),
        'allocation_note': allocation_note, 'plan_sha256': EXPECTED_CONTROL_PLAN,
        'preparation_sha256': control.sha(PREPARATION), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
    state = {'status': 'waiting_NeXT_full5000', 'identity': predecessor.identity(os.getpid()),
        'registered_utc': utc(), 'allocation_note': allocation_note, 'purpose': PURPOSE,
        'GPU_slot': 'B', 'GPU_loaded_by_supervisor': False, 'run_plan_sha256': EXPECTED_CONTROL_PLAN}
    save('supervisor_state.json', state)
    print(json.dumps(state), flush=True)
    previous_reason = None
    while True:
        ready, gate = predecessor.prerequisites()
        if gate['reason'] != previous_reason:
            state.update(gate=gate, observed_utc=utc())
            save('supervisor_state.json', state)
            print(json.dumps({'event': 'gate', **gate}), flush=True)
            previous_reason = gate['reason']
        if ready:
            break
        time.sleep(5)
    frozen_gate()
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(('USE_LLAVA_ARCH_', 'ANCHORZIP_', 'ENTROPY_', 'EADP_')):
            del env[name]
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    state.update(status='independent_CPU_predecessor_verification', observed_utc=utc())
    save('supervisor_state.json', state)
    with (RUN / 'predecessor_verification.log').open('w') as log:
        verified = subprocess.run([PYTHON, str(ROOT / 'verify_next_predecessor.py')], env=env,
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    assert verified.returncode == 0, 'Independent predecessor verification failed; no GPU launch'
    validate_independent_predecessor()
    previous_reason = None
    while True:
        ready, capacity = capacity_gate()
        if capacity['reason'] != previous_reason:
            state.update(status='waiting_explicit_slot_B_capacity', capacity=capacity, observed_utc=utc())
            save('supervisor_state.json', state)
            print(json.dumps({'event': 'capacity', **capacity}), flush=True)
            previous_reason = capacity['reason']
        if ready:
            break
        time.sleep(5)
    frozen_gate()
    validate_independent_predecessor()
    ready, capacity = capacity_gate()
    assert ready
    exclusive_json(RUN / 'one_launch_attempt.json', {'attempt': 1, 'utc': utc(), 'automatic_retry': False,
        'allocation_note': allocation_note, 'control_plan_sha256': EXPECTED_CONTROL_PLAN})
    save('launch_gate_evidence.json', {'verified_utc': utc(), 'allocation_note': allocation_note,
        'control_plan_sha256': EXPECTED_CONTROL_PLAN, 'preparation_sha256': control.sha(PREPARATION),
        'independent_predecessor_verification': {'path': str(predecessor.OUTPUT), 'sha256': control.sha(predecessor.OUTPUT)},
        'capacity': capacity, 'no_score_direction_threshold': True})
    command = [PYTHON, str(ROOT / 'alpha_control.py'), 'run', '--gpu-authorized', '--allocation-note', allocation_note]
    env['CUDA_VISIBLE_DEVICES'] = '0'
    with (RUN / 'supervised_run.log').open('w') as log:
        controller = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        state.update(status='control_started', launched_utc=utc(), controller=predecessor.identity(controller.pid), command=command)
        save('supervisor_state.json', state)
        print(json.dumps({'event': 'control_started', **state}), flush=True)
        first_seen, worker_seen = False, False
        while controller.poll() is None:
            if not worker_seen and (RUN / 'process_identity.json').exists():
                try:
                    worker = read(RUN / 'process_identity.json')
                except json.JSONDecodeError:
                    worker = None
                if worker:
                    state['actual_worker'] = worker
                    save('supervisor_state.json', state)
                    print(json.dumps({'event': 'actual_worker_started', 'worker': worker}), flush=True)
                    worker_seen = True
            prediction = RUN / 'predictions.jsonl'
            if not first_seen and prediction.exists() and prediction.stat().st_size:
                with prediction.open() as stream:
                    line = stream.readline()
                if line.endswith('\n'):
                    state['first_prediction'] = json.loads(line)
                    save('supervisor_state.json', state)
                    print(json.dumps({'event': 'first_prediction', 'prediction': state['first_prediction']}), flush=True)
                    first_seen = True
            time.sleep(2)
        exit_code = controller.wait()
    assert exit_code == 0, f'POPE controller exit{exit_code}; no retry'
    release = complete_release()
    state.update(status='complete', exit_code=0, completed_utc=utc(), actual_worker_exited=True,
        full_result_summary_sha256=release['full_result_summary']['sha256'],
        completion_and_release_proof_sha256=control.sha(RUN / 'completion_and_release_proof.json'))
    save('supervisor_state.json', state)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='check', choices=['prepare', 'check', 'run'])
    parser.add_argument('--root-allocated-slot-B', action='store_true')
    parser.add_argument('--allocation-note')
    args = parser.parse_args()
    try:
        if args.action == 'prepare':
            print(json.dumps(prepare()))
        elif args.action == 'check':
            frozen_gate()
            ready, gate = predecessor.prerequisites()
            print(json.dumps({'CPU_only': True, 'would_launch': False, 'gate_passed': ready, 'gate': gate}))
        else:
            assert args.root_allocated_slot_B and args.allocation_note, 'Future explicit root slot B allocation required'
            run(args.allocation_note)
    except BaseException as error:
        if args.action == 'run' and CLAIM.exists():
            save('supervisor_failure.json', {'status': 'terminal_failure_no_retry', 'utc': utc(),
                'error': repr(error), 'identity': predecessor.identity(os.getpid()), 'automatic_retry': False,
                'preserve_all_outputs': True})
        raise
