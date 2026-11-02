#!/usr/bin/env python3
"""CPU-only slot-B waiter. Preparation/check never arms or launches a model."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import run_beta2 as beta

OUT = Path(__file__).resolve().parent
POPE = OUT.parents[1] / 'pope/run03_single_alpha05_beta1_wait_k32'
POPE_PLAN_SHA = 'bf40e511c27cf920dbd287d61126c405346f97fc37728cff3254af608aef5a70'
BETA_PLAN_SHA = 'f4a2eff7485313e8ee1bec7553c4897b60abb48ccd4beb34e165e20a09e49f40'
BETA_WORKER_SHA = '19c3865545d650503e286ce37ecf1803cd39d6d98ce28b534173c76ea2294232'
PYTHON = '/home/dell/miniconda3/envs/llava_pruner/bin/python'
PREFIX = 'after_pope_alpha'
read = lambda path: json.loads(Path(path).read_text())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
now = lambda: datetime.now(timezone.utc).isoformat()


def save(name, record):
    target = OUT / (PREFIX + '.' + name + '.json')
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)


def process_identity(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': int(pid), 'start_ticks': int(fields[19]), 'state': fields[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def gpu_apps():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True)
    return output, {int(line.split(',', 1)[0]) for line in output.splitlines() if line.strip()}


def beta_frozen_gate():
    assert sha(OUT / 'protocol.json') == BETA_PLAN_SHA
    assert read(OUT / 'READY.json')['protocol_sha256'] == BETA_PLAN_SHA
    assert sha(OUT / 'worker.py') == BETA_WORKER_SHA
    beta.check_plan(read(OUT / 'protocol.json'))
    assert 'torch' not in sys.modules, 'CPU waiter must not import torch/model'


def pope_frozen_gate(plan, hash_images=False):
    assert sha(POPE / 'run_plan.json') == POPE_PLAN_SHA
    assert plan['run_id'] == POPE.name and (plan['parameters']['alpha'], plan['parameters']['beta'], plan['parameters']['visual_token_num']) == (.5, 1., 32)
    for name, expected in plan['source_files'].items():
        assert sha(Path(plan['source_root']) / name) == expected, name
    for group in ('small_file_guards', 'implementation_sha256', 'model_small_files'):
        for name, expected in plan[group].items():
            assert sha(name) == expected, name
    for group in ('model_stat_guards', 'image_stat_guards'):
        for name, guard in plan[group].items():
            path = Path(name)
            stat = path.stat()
            actual = {'resolved': str(path.resolve()), 'bytes': stat.st_size, 'device': stat.st_dev,
                      'inode': stat.st_ino, 'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns}
            assert actual == {k: v for k, v in guard.items() if k != 'prior_verified_sha256'}, name
            if group == 'image_stat_guards' and hash_images:
                assert sha(path) == guard['prior_verified_sha256'], name
    return {'source_hashes': len(plan['source_files']), 'image_hashes_rechecked': len(plan['image_stat_guards']) if hash_images else 0,
            'model_payload_prior_SHA_and_current_stat_guards': len(plan['model_stat_guards'])}


def terminal_status(status):
    value = str(status).lower()
    if 'fail' in value or value in ('error', 'aborted', 'cancelled'):
        raise RuntimeError('Predecessor terminal failure; preserve outputs, no retry: ' + value)
    return value == 'complete'


def exclusive_record(path, record):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, indent=2) + '\n')


def predecessor_liveness(state, supervisor):
    """Process exit is evidence; elapsed observation time never implies failure."""
    if supervisor and supervisor.get('status') != 'complete' and 'identity' in supervisor:
        original = supervisor['identity']
        current = process_identity(original['pid'])
        if not current or current['start_ticks'] != original['startticks']:
            refreshed = read(POPE / 'supervisor_state.json')
            terminal_status(refreshed['status'])
            if refreshed['status'] != 'complete':
                raise RuntimeError('Original POPE supervisor exited without terminal completion; preserve model/output state for root review')
    identity_path = POPE / 'process_identity.json'
    if state['status'] != 'generating' or not identity_path.exists():
        return
    original = read(identity_path)
    current = process_identity(original['worker_pid'])
    if current and current['start_ticks'] == original['worker_start_ticks']:
        return
    # A normal worker.wait() -> validating transition may briefly retain generating.
    refreshed = read(POPE / 'state.json')
    terminal_status(refreshed['status'])
    if refreshed['status'] != 'generating':
        return
    controller = supervisor.get('controller') if supervisor else None
    if controller:
        assert controller['pid'] == original['controller_pid']
        current_controller = process_identity(controller['pid'])
        if not current_controller or current_controller['start_ticks'] != controller['startticks']:
            raise RuntimeError('Original POPE worker and controller exited while state still says generating; no further completion is possible')
    elif process_identity(original['controller_pid']) is None:
        raise RuntimeError('POPE worker and recorded controller PID are absent while state still says generating')


def independent_pope(plan, summary):
    raw_path = POPE / 'predictions.jsonl'
    raw = raw_path.read_bytes()
    assert raw.endswith(b'\n')
    answers = [json.loads(line) for line in raw.splitlines()]
    question_path = Path(plan['generation_command'][plan['generation_command'].index('--question-file') + 1])
    questions = [json.loads(line) for line in question_path.read_text().splitlines()]
    assert len(answers) == len(questions) == 8910
    assert len({a['question_id'] for a in answers}) == 8910
    assert [a['question_id'] for a in answers] == [q['question_id'] for q in questions]
    assert all(a['prompt'] == q['text'] and a['model_id'] == 'llava-v1.5-7b' and isinstance(a['text'], str)
               and not a['text'].startswith('FAILED') for a, q in zip(answers, questions))
    label_dir = Path(plan['score_command'][plan['score_command'].index('--annotation-dir') + 1])
    def positive(text):
        words = text.split('.')[0].replace(',', '').split(' ')
        return not any(word in words for word in ('No', 'not', 'no'))
    categories = {}
    for category, expected in (('random', 2910), ('popular', 3000), ('adversarial', 3000)):
        selected = [(a, q) for a, q in zip(answers, questions) if q['category'] == category]
        labels = [json.loads(line) for line in (label_dir / ('coco_pope_' + category + '.json')).read_text().splitlines()]
        assert len(selected) == len(labels) == expected
        for (_, question), label in zip(selected, labels):
            assert question['image'] == label['image']
            assert question['text'].replace('\nAnswer the question using a single word or phrase.', '') == label['text']
        counts = Counter((positive(a['text']), label['label'] != 'no') for (a, _), label in zip(selected, labels))
        tp, fp, tn, fn = (counts[key] for key in ((True, True), (True, False), (False, False), (False, True)))
        f1 = 200 * tp / (2 * tp + fp + fn)
        actual = summary['categories'][category]
        assert actual['n'] == expected and [actual[k] for k in ('TP', 'FP', 'TN', 'FN')] == [tp, fp, tn, fn]
        assert abs(actual['f1_percent'] - f1) < 1e-10
        categories[category] = f1
    independent = sum(categories.values()) / 3
    log_path = Path(plan['paths']['score_log'])
    assert sha(log_path) == summary['score_log_sha256']
    author = float(re.search(r'Average F1 score:\s*([0-9.eE+-]+)', log_path.read_text()).group(1)) * 100
    assert math.isfinite(author) and math.isfinite(independent)
    assert abs(author - independent) < 1e-10
    assert abs(author - summary['official_macro_f1_percent']) < 1e-10
    assert abs(independent - summary['independent_macro_f1_percent']) < 1e-10
    assert sha(raw_path) == summary['predictions_sha256']
    return {'n_predictions': 8910, 'ordered_IDs_prompts_and_pinned_labels_verified': True,
            'official_macro_F1_percent': author, 'independent_macro_F1_percent': independent,
            'predictions_sha256': sha(raw_path), 'official_score_log_sha256': sha(log_path)}


def predecessor_gate():
    if (POPE / 'supervisor_failure.json').exists():
        raise RuntimeError('POPE supervisor failed before or after model launch; preserve outputs, no retry')
    supervisor_path = POPE / 'supervisor_state.json'
    supervisor = None
    if supervisor_path.exists():
        supervisor = read(supervisor_path)
        terminal_status(supervisor['status'])
        assert supervisor.get('exit_code') in (None, 0), 'POPE supervisor exited nonzero'
    plan = read(POPE / 'run_plan.json')
    assert sha(POPE / 'run_plan.json') == POPE_PLAN_SHA
    state = read(POPE / 'state.json')
    predecessor_liveness(state, supervisor)
    if not terminal_status(state['status']):
        return False, {'reason': 'waiting_POPE_completion', 'predecessor_status': state['status']}
    names = ['process_identity.json', 'full_result_summary.json', 'completion_and_release_proof.json', 'supervisor_state.json']
    if any(not (POPE / name).exists() for name in names):
        return False, {'reason': 'waiting_POPE_final_summary_and_release'}
    identity, summary, release, supervisor = (read(POPE / name) for name in names)
    if not terminal_status(supervisor['status']):
        return False, {'reason': 'waiting_POPE_supervisor_finalization'}
    assert supervisor['exit_code'] == 0 and supervisor['actual_worker_exited']
    assert supervisor['run_plan_sha256'] == POPE_PLAN_SHA
    assert supervisor['full_result_summary_sha256'] == sha(POPE / 'full_result_summary.json')
    assert supervisor['completion_and_release_proof_sha256'] == sha(POPE / 'completion_and_release_proof.json')
    assert state['generation_exit_code'] == state['score_exit_code'] == 0 and state['n_predictions'] == 8910
    assert state['plan_sha256'] == identity['plan_sha256'] == summary['run_plan_sha256'] == POPE_PLAN_SHA
    assert identity['generation_command'] == plan['generation_command']
    assert state['worker_pid'] == identity['worker_pid']
    assert summary['complete'] and summary['n_predictions'] == 8910 and summary['official_scorer_exit_code'] == 0
    for key in ('all_question_ids_order_verified', 'all_raw_prompts_verified', 'all_pinned_image_question_positions_verified',
                'gpu_worker_exit_verified', 'frozen_source_inputs_payload_stats_verified', 'actual_runtime_verified'):
        assert summary[key], key
    assert (summary['actual_alpha'], summary['actual_beta'], summary['actual_visual_token_num']) == (.5, 1., 32)
    assert release['status'] == 'all_completion_and_exit_gates_passed'
    assert release['GPU_slot'] == 'B' and release['control_run'] == POPE.name
    assert release['next_owner'] in ('/root/v2_review', 'MMB beta2', 'mmbench/beta2_control')
    assert release['worker_pid'] == identity['worker_pid'] and release['worker_start_ticks'] == identity['worker_start_ticks']
    assert not release['same_worker_still_alive']
    assert release['generation_exit_code'] == release['official_score_exit_code'] == 0 and release['prediction_count'] == 8910
    assert release['independent_and_official_macro_match'] and release['all_source_sha_match']
    assert release['gpu_pid_absent'] and release['all_frozen_source_input_weight_image_guards_verified']
    assert release['run_plan_sha256'] == POPE_PLAN_SHA
    assert release['state_sha256'] == sha(POPE / 'state.json')
    assert release['process_identity_sha256'] == sha(POPE / 'process_identity.json')
    assert release['official_score_log_sha256'] == summary['score_log_sha256']
    separate = release['independent_prediction_verification']
    separate_path = POPE / 'supervisor_prediction_verification/full_prediction_verification.json'
    assert Path(separate['path']).resolve() == separate_path.resolve() and separate['sha256'] == sha(separate_path)
    separate_result = read(separate_path)
    assert separate_result['complete'] and separate_result['n_predictions'] == 8910
    assert separate_result['predictions_sha256'] == summary['predictions_sha256']
    assert abs(separate_result['official_macro_f1_percent'] - summary['official_macro_f1_percent']) < 1e-10
    assert abs(separate_result['independent_macro_f1_percent'] - summary['independent_macro_f1_percent']) < 1e-10
    assert Path(release['full_result_summary']['path']).resolve() == (POPE / 'full_result_summary.json').resolve()
    assert release['full_result_summary']['sha256'] == sha(POPE / 'full_result_summary.json')
    assert release['predictions_sha256'] == summary['predictions_sha256']
    current = process_identity(identity['worker_pid'])
    if current and current['start_ticks'] == identity['worker_start_ticks']:
        return False, {'reason': 'waiting_original_POPE_worker_exit', 'identity': current}
    apps, pids = gpu_apps()
    assert identity['worker_pid'] not in pids, 'Released POPE PID is still listed as a GPU app'
    if len(pids) >= 2:
        return False, {'reason': 'waiting_global_model_count_below2', 'GPU_processes': apps}
    frozen = pope_frozen_gate(plan, hash_images=True)
    scores = independent_pope(plan, summary)
    return True, {'reason': 'all_POPE_integrity_score_and_exit_gates_passed', 'verified_utc': now(),
                  'predecessor_files': {name: {'path': str(POPE / name), 'sha256': sha(POPE / name)} for name in ['run_plan.json', 'state.json', *names]},
                  'worker_identity': identity, 'original_worker_exited': True, 'current_identity_if_PID_reused': current,
                  'GPU_processes_before_launch': apps, 'frozen_guards': frozen, 'full_Pope_score_check': scores,
                  'score_direction_used_as_gate': False, 'repeat_consistency_used_as_gate': False}


def main(arm, allocation_note):
    beta_frozen_gate()
    pope_frozen_gate(read(POPE / 'run_plan.json'))
    assert not (OUT / 'state.json').exists(), 'Never restart an existing beta2 run'
    if not arm:
        try:
            passed, evidence = predecessor_gate()
        except json.JSONDecodeError:
            passed, evidence = False, {'reason': 'waiting_atomic_predecessor_write'}
        print(json.dumps({'CPU_only': True, 'would_launch': False, 'gate_passed': passed, 'gate': evidence}, ensure_ascii=False))
        return
    assert allocation_note, 'Root must explicitly allocate slot B before arming'
    assert not (OUT / (PREFIX + '.registration.json')).exists(), 'No automatic supervisor restart'
    command = [PYTHON, str(OUT / 'run_beta2.py'), 'run', '--allocation', allocation_note]
    registration = {'registered_utc': now(), 'identity': process_identity(os.getpid()), 'GPU_slot': 'B',
                    'root_allocation_note': allocation_note, 'command': command, 'supervisor_sha256': sha(__file__),
                    'beta_plan_sha256': BETA_PLAN_SHA, 'POPE_plan_sha256': POPE_PLAN_SHA,
                    'GPU_loaded_by_waiter': False, 'automatic_retry': False, 'global_max_models_including_loading': 2}
    exclusive_record(OUT / (PREFIX + '.claim.json'), registration)
    save('registration', registration)
    state = {'status': 'waiting_POPE_complete_and_exit', **registration}
    save('state', state)
    print(json.dumps(state), flush=True)
    previous = None
    while True:
        try:
            passed, evidence = predecessor_gate()
        except json.JSONDecodeError:
            passed, evidence = False, {'reason': 'waiting_atomic_predecessor_write'}
        if evidence['reason'] != previous:
            state.update(gate=evidence, observed_utc=now())
            save('state', state)
            print(json.dumps({'event': 'gate', **evidence}), flush=True)
            previous = evidence['reason']
        if passed:
            break
        time.sleep(10)
    beta_frozen_gate()
    assert sha(__file__) == registration['supervisor_sha256']
    passed, evidence = predecessor_gate()
    assert passed
    save('launch_gate', {'root_allocation_note': allocation_note, 'GPU_slot': 'B', **evidence})
    exclusive_record(OUT / (PREFIX + '.one_launch_attempt.json'), {'attempt': 1, 'recorded_utc': now(),
                     'root_allocation_note': allocation_note, 'GPU_slot': 'B', 'automatic_retry': False,
                     'registration_sha256': sha(OUT / (PREFIX + '.registration.json'))})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
    with (OUT / (PREFIX + '.generation.log')).open('x') as log:
        worker = subprocess.Popen(command, cwd=OUT.parents[3], env=env, stdout=log, stderr=subprocess.STDOUT)
        state.update(status='beta2_started', launched_utc=now(), run_controller=process_identity(worker.pid))
        save('state', state)
        print(json.dumps({'event': 'beta2_controller_started', 'identity': state['run_controller']}), flush=True)
        code = worker.wait()
    state.update(status='complete' if code == 0 else 'terminal_failure_no_retry', exit_code=code, finished_utc=now())
    if code == 0:
        result = read(OUT / 'independent_verification.json')
        assert result['success'] and result['n_rows'] == 4377 and result['n_circular_groups'] == 1176
        beta_frozen_gate()
        native = read(OUT / 'state.json')['native_process']
        current = process_identity(native['pid'])
        assert not current or current['start_ticks'] != native['start_ticks']
        _, pids = gpu_apps()
        assert native['pid'] not in pids
        state.update(independent_verification_sha256=sha(OUT / 'independent_verification.json'), actual_native_exited=True)
    save('state', state)
    print(json.dumps(state), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', action='store_true')
    parser.add_argument('--allocation-note')
    args = parser.parse_args()
    try:
        main(args.arm, args.allocation_note)
    except Exception as error:
        if args.arm:
            save('failure', {'status': 'terminal_failure_no_retry', 'utc': now(), 'error': repr(error), 'identity': process_identity(os.getpid()), 'automatic_retry': False})
        raise
