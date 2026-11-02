#!/usr/bin/env python3
"""CPU-only independent terminal gate for the exact NeXT predecessor."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
NEXT = ROOT.parent / 'next_textvqa'
RUN = NEXT / 'full_default_K128_alpha05_beta1_wait'
PLAN = NEXT / 'full_default_K128.plan.json'
WORKER = NEXT / 'repro_full_default.py'
EXPECTED_PLAN = '7de9048ad555683ca95d1bba27193c30c641b2decd5ae2d4626a359b11e9462d'
EXPECTED_WORKER = '466e3c7be72cf5db3a12409e50fcb47286f6d054af4fa6f6f0fad351a6d4e94d'
PID, TICKS = 2399640, 1331554245
SUPERVISOR_PID, SUPERVISOR_TICKS = 2388364, 1331504632
OUTPUT = ROOT / 'run03_single_alpha05_beta1_wait_k32/predecessor_verification.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return {'pid': pid, 'startticks': int(fields[19])}


def gpu_apps():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True)
    processes = []
    for line in output.splitlines():
        if line.strip():
            processes.append({'pid': int(line.split(',', 1)[0]), 'line': line})
    return output, processes


def check_frozen_next():
    assert sha(PLAN) == EXPECTED_PLAN and sha(WORKER) == EXPECTED_WORKER
    spec = importlib.util.spec_from_file_location('frozen_next_full_verification', WORKER)
    full = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(full)
    full.check_plan(read(PLAN))
    return full


def prerequisites():
    """Small terminal files first; do not read running per-question traces."""
    assert sha(PLAN) == EXPECTED_PLAN and sha(WORKER) == EXPECTED_WORKER
    if (RUN / 'failure.json').exists():
        raise RuntimeError('NeXT failure.json exists; no retry or slot takeover')
    if (NEXT / 'full_default_K128.supervisor_failure.json').exists():
        raise RuntimeError('NeXT supervisor_failure.json exists; preserve completed worker artifacts, no retry')
    paths = {'state': RUN / 'state.json', 'supervisor': NEXT / 'full_default_K128.supervisor_state.json', 'score': RUN / 'official.score.json'}
    available = {}
    try:
        for name, path in paths.items():
            if path.exists():
                available[name] = read(path)
    except json.JSONDecodeError:
        worker_gone = identity(PID) != {'pid': PID, 'startticks': TICKS}
        supervisor_gone = identity(SUPERVISOR_PID) != {'pid': SUPERVISOR_PID, 'startticks': SUPERVISOR_TICKS}
        if not (worker_gone and supervisor_gone):
            return False, {'reason': 'waiting_for_atomic_terminal_records'}
        try:
            available = {name: read(path) for name, path in paths.items() if path.exists()}
        except json.JSONDecodeError as error:
            raise RuntimeError('NeXT worker/controller exited with malformed terminal record after re-read') from error
    for name, record in available.items():
        status = str(record.get('status', '')).lower()
        if any(word in status for word in ['fail', 'error', 'cancel', 'abort']):
            raise RuntimeError('NeXT terminal failure: ' + name + ' ' + status)
        if record.get('exit_code') not in [None, 0]:
            raise RuntimeError('NeXT predecessor nonzero exit')
    current = identity(PID)
    original_alive = current is not None and current['startticks'] == TICKS
    state_complete = available.get('state', {}).get('status') == 'complete'
    supervisor_complete = available.get('supervisor', {}).get('status') == 'complete'
    if state_complete and not original_alive and not supervisor_complete:
        supervisor_identity = identity(SUPERVISOR_PID)
        if supervisor_identity != {'pid': SUPERVISOR_PID, 'startticks': SUPERVISOR_TICKS}:
            # The controller may have written complete between our first read
            # and its exit. Re-read terminal records before declaring failure.
            try:
                available = {name: read(path) for name, path in paths.items() if path.exists()}
            except json.JSONDecodeError as error:
                raise RuntimeError('NeXT controller exited with malformed terminal record; preserve worker/raw') from error
            if (NEXT / 'full_default_K128.supervisor_failure.json').exists():
                raise RuntimeError('NeXT supervisor failure recorded during terminal re-read')
            if (len(available) != len(paths) or available.get('state', {}).get('status') != 'complete'
                    or available.get('supervisor', {}).get('status') != 'complete'):
                raise RuntimeError('NeXT worker completed/exited but original supervisor exited without complete terminal state; preserve worker/raw')
    if len(available) < len(paths):
        if not original_alive and available.get('state', {}).get('status') != 'complete':
            raise RuntimeError('NeXT worker exited without complete5000 terminal records')
        return False, {'reason': 'waiting_full5000_official_terminal_records', 'present': list(available)}
    state, supervisor, score = (available[key] for key in ['state', 'supervisor', 'score'])
    assert supervisor['identity'] == {'pid': SUPERVISOR_PID, 'startticks': SUPERVISOR_TICKS}
    if state['status'] != 'complete' or supervisor['status'] != 'complete':
        if not original_alive and state['status'] != 'complete':
            raise RuntimeError('NeXT worker exited before complete terminal state')
        return False, {'reason': 'waiting_predecessor_complete_states'}
    assert state['n'] == score['n'] == score['token_decoding_checks'] == 5000
    assert state['pid'] == PID and supervisor['worker'] == {'pid': PID, 'startticks': TICKS}
    assert supervisor['exit_code'] == 0 and supervisor['actual_worker_exited'] is True
    assert supervisor['plan_sha256'] == EXPECTED_PLAN and supervisor['worker_sha256'] == EXPECTED_WORKER
    assert state['score_sha256'] == supervisor['score_sha256'] == sha(paths['score'])
    assert score['plan']['sha256'] == EXPECTED_PLAN
    if original_alive:
        return False, {'reason': 'waiting_actual_original_NeXT_worker_exit'}
    apps, processes = gpu_apps()
    if any(process['pid'] == PID for process in processes):
        return False, {'reason': 'waiting_predecessor_absent_from_nvidia_smi', 'nvidia_smi': apps}
    return True, {'reason': 'terminal5000_and_actual_worker_exit_ready_for_independent_CPU_verification',
        'state_sha256': sha(paths['state']), 'supervisor_sha256': sha(paths['supervisor']),
        'official_score_sha256': sha(paths['score']), 'current_identity_if_reused': current, 'nvidia_smi': apps}


def consensus_from_count(match_count):
    """Independent closed form of ten-annotator leave-one-out VQA consensus."""
    assert type(match_count) is int and 0 <= match_count <= 10
    return (match_count * min(max(match_count - 1, 0) / 3, 1) + (10 - match_count) * min(match_count / 3, 1)) / 10


def verify(output_path=OUTPUT):
    ready, gate = prerequisites()
    assert ready, gate
    full = check_frozen_next()
    recomputed = full.verify(write=False)
    assert recomputed['n'] == recomputed['token_decoding_checks'] == 5000
    official_path = RUN / 'official.score.json'
    official = read(official_path)
    assert abs(recomputed['value'] - official['value']) < 1e-10
    plan = read(PLAN)
    questions = full.rows(plan['questions']['path'])
    answers = full.rows(RUN / 'answers.jsonl')
    official_rows = full.rows(RUN / 'per_question_scores.jsonl')
    assert len(questions) == len(answers) == len(official_rows) == 5000
    assert sha(RUN / 'per_question_scores.jsonl') == official['per_question_scores_sha256']
    prompt_processor, _ = full.author_imports()
    from llava.eval.m4c_evaluator import EvalAIAnswerProcessor
    normalizer = EvalAIAnswerProcessor()
    annotations = read(plan['annotation']['path'])['data']
    by_key = {(str(row['image_id']), row['question'].lower()): row for row in annotations}
    assert len(by_key) == len(annotations) == 5000
    independent, max_delta = [], 0.
    for index, (question, answer, scored) in enumerate(zip(questions, answers, official_rows)):
        assert scored['position'] == index and str(scored['question_id']) == str(question['question_id'])
        assert scored['image'] == question['image']
        assert scored['prompt_sha256'] == hashlib.sha256(question['text'].encode()).hexdigest()
        gt = by_key[(str(question['question_id']), prompt_processor(question['text']))]['answers']
        assert len(gt) == 10
        predicted = normalizer(answer['text'])
        matches = sum(normalizer(value) == predicted for value in gt)
        value = consensus_from_count(matches)
        delta = abs(value - scored['score'])
        assert delta < 1e-12
        max_delta = max(max_delta, delta)
        independent.append(value)
    accuracy = 100 * sum(independent) / 5000
    assert abs(accuracy - official['value']) < 1e-10
    for name, artifact in official['artifacts'].items():
        assert sha(artifact['path']) == artifact['sha256'] == recomputed['artifacts'][name]['sha256']
    ready, exit_gate = prerequisites()
    assert ready
    assert not sys.modules['torch'].cuda.is_initialized()
    result = {'success': True, 'checked_utc': datetime.now(timezone.utc).isoformat(),
        'n_rows': 5000, 'token_decoding_checks': 5000, 'per_question_score_checks': 5000,
        'official_accuracy_percent': official['value'], 'independent_accuracy_percent': accuracy,
        'max_per_question_score_delta': max_delta, 'official_score_sha256': sha(official_path),
        'plan_sha256': EXPECTED_PLAN, 'worker_sha256': EXPECTED_WORKER,
        'artifacts': {**official['artifacts'], 'per_question_scores.jsonl': {
            'path': str(RUN / 'per_question_scores.jsonl'), 'sha256': sha(RUN / 'per_question_scores.jsonl')}},
        'all_frozen_source_input_weight_image_guards_verified': True,
        'source_sha256': {row['author_path']: row['author_sha256'] for row in plan['source_comparisons']},
        'questions': plan['questions'], 'annotation': plan['annotation'],
        'worker_exit': {'pid': PID, 'startticks': TICKS, 'original_worker_exited': True, 'gpu_pid_absent': True,
            'current_identity_if_reused': identity(PID), 'nvidia_smi': exit_gate['nvidia_smi']},
        'predecessor_state_sha256': exit_gate['state_sha256'], 'predecessor_supervisor_sha256': exit_gate['supervisor_sha256'],
        'independent_scoring_method': 'Reuse canonical author EvalAI normalization; independently derive normalizedGT match count and closed-form ten-annotator leave-one-out consensus for all5000. Compare each saved official score and aggregate; native verifier separately decodes captured output IDs.',
        'score_direction_threshold_or_paper_closeness_used': False, 'cuda_initialized': False,
        'verifier': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)}}
    temporary = output_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(output_path)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        ready, gate = prerequisites()
        print(json.dumps({'CPU_only': True, 'would_launch': False, 'gate_passed': ready, 'gate': gate}))
    else:
        print(json.dumps(verify()))
