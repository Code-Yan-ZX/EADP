"""CPU-only one-shot release evidence after the frozen full NeXT run exits.

This observes the existing supervisor; it cannot launch a model or retry a run.
Metric independence is provided separately by the POPE predecessor audit.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import repro_full_default as full

HERE = Path(__file__).resolve().parent
SUPERVISOR = HERE / 'full_default_K128.supervisor_state.json'
PLAN_SHA = '7de9048ad555683ca95d1bba27193c30c641b2decd5ae2d4626a359b11e9462d'
WORKER_SHA = '466e3c7be72cf5db3a12409e50fcb47286f6d054af4fa6f6f0fad351a6d4e94d'
PID, TICKS = 2399640, 1331554245


def utc():
    return datetime.now(timezone.utc).isoformat()


def identity(pid):
    try:
        text = Path('/proc', str(pid), 'stat').read_text()
        return {'pid': pid, 'startticks': int(text[text.rindex(')') + 2:].split()[19])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def read(path):
    return json.loads(Path(path).read_text())


def gate():
    assert 'torch' not in sys.modules
    if (HERE / 'full_default_K128.supervisor_failure.json').exists() or (full.RUN / 'failure.json').exists():
        raise RuntimeError('Preserved terminal failure; no release certification or retry')
    try:
        state = read(SUPERVISOR)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if state['status'] == 'terminal_failure_no_retry':
        raise RuntimeError('Original supervisor reports failure')
    if state['status'] != 'complete':
        return None
    assert state['exit_code'] == 0 and state['actual_worker_exited']
    assert state['worker'] == {'pid': PID, 'startticks': TICKS}
    assert full.sha(full.PLAN) == PLAN_SHA and full.sha(full.__file__) == WORKER_SHA
    plan = read(full.PLAN)
    full.check_plan(plan)
    assert full.sha(HERE / 'supervise_after_mmbench.py') == state['supervisor_sha256']
    native = read(full.RUN / 'state.json')
    score_path = full.RUN / 'official.score.json'
    score = read(score_path)
    score_sha = full.sha(score_path)
    assert native['status'] == 'complete' and native['pid'] == PID and native['n'] == 5000
    assert score_sha == native['score_sha256'] == state['score_sha256']
    assert score['n'] == score['token_decoding_checks'] == 5000
    assert score['plan']['sha256'] == PLAN_SHA
    for record in score['artifacts'].values():
        assert full.sha(record['path']) == record['sha256']
    assert full.sha(full.RUN / 'per_question_scores.jsonl') == score['per_question_scores_sha256']
    current = identity(PID)
    if current and current['startticks'] == TICKS:
        return None
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                    '--format=csv,noheader'], text=True)
    if any(int(line.split(',', 1)[0]) == PID for line in apps.splitlines() if line.strip()):
        return None
    return {'checked_utc': utc(), 'CPU_only': True, 'GPU_loaded': False,
            'worker': {'pid': PID, 'startticks': TICKS}, 'original_worker_exited': True,
            'gpu_pid_absent': True, 'current_identity_if_pid_reused': current,
            'nvidia_smi': apps, 'official_score_sha256': score_sha,
            'supervisor_sha256': full.sha(SUPERVISOR), 'native_state_sha256': full.sha(full.RUN / 'state.json'),
            'plan_sha256': PLAN_SHA, 'worker_sha256': WORKER_SHA,
            'finalizer_sha256': full.sha(__file__), 'all_frozen_guards_passed': True,
            'n_rows': 5000, 'token_decoding_checks': 5000,
            'official_accuracy_percent': score['value'],
            'limits': 'Process exit and frozen artifact certification only; independent metric audit is a separate report.'}


def main(check_only):
    if check_only:
        result = gate()
        print(json.dumps({'CPU_only': True, 'would_launch_GPU': False, 'terminal_release_ready': result is not None}))
        return
    state_path = full.RUN / 'release_supervisor_state.json'
    assert not state_path.exists(), 'Never restart this finalizer automatically'
    registration = {'status': 'waiting_full_terminal', 'registered_utc': utc(),
                    'identity': identity(os.getpid()), 'CPU_only': True, 'GPU_loaded': False,
                    'script_sha256': full.sha(__file__), 'worker': {'pid': PID, 'startticks': TICKS}}
    full.save(state_path, registration)
    while True:
        result = gate()
        if result is not None:
            full.save(full.RUN / 'release.json', result)
            full.save(state_path, dict(registration, status='complete', finished_utc=utc(),
                                      release_sha256=full.sha(full.RUN / 'release.json')))
            print(json.dumps({'status': 'complete', 'GPU_loaded': False}), flush=True)
            return
        time.sleep(15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    try:
        main(args.check_only)
    except BaseException as error:
        if not args.check_only:
            full.save(full.RUN / 'release_supervisor_failure.json',
                      {'status': 'terminal_failure_no_retry', 'utc': utc(), 'error': repr(error), 'GPU_loaded': False})
        raise
