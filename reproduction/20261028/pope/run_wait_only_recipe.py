#!/usr/bin/env python3
"""Registered same-recipe wait-only control; never overlaps its source baseline."""
import argparse
import ast
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from wait_only_patch import build

OUT = Path(__file__).resolve().parent
RUN01 = OUT / 'run01_source_as_is_k32'
RUN02 = OUT / 'run02_source_wait_only_k32'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def prepare():
    RUN02.mkdir(exist_ok=True)
    base = json.loads((RUN01 / 'run_plan.json').read_text())
    plan = copy.deepcopy(base)
    paths = {name: RUN02 / Path(value).name for name, value in base['paths'].items()}
    command = [base['generation_command'][0], str(OUT / 'worker_wait_only.py')] + base['generation_command'][3:]
    command[command.index('--answers-file') + 1] = str(paths['predictions'])
    score = base['score_command'][:]
    score[score.index('--result-file') + 1] = str(paths['predictions'])
    patched_ast, proof = build()
    save(RUN02 / 'prepared_patch_proof.json', proof)
    (RUN02 / 'declared_patched_forward.py').write_text(ast.unparse(patched_ast) + '\n')
    plan.update(run_id=RUN02.name, recipe_origin='same released POPE shell alpha1/beta1/K32 recipe, with declared three-node caller-stream wait only',
        generation_command=command, score_command=score, source_runtime_patches=proof['inserted_nodes'],
        paths={k: str(v) for k, v in paths.items()}, base_plan={'path': str(RUN01 / 'run_plan.json'), 'sha256': sha(RUN01 / 'run_plan.json')},
        trigger={'observed_same_input_yes_no_conflicts': 12, 'observed_repeated_groups': 179, 'observed_prefix_rows': 3328,
                 'diagnostic': str(RUN01 / 'same_input_consistency.json'), 'no_prefix_accuracy_used': True},
        launch_gate=['source baseline all8910 generated in canonical order', 'source baseline official scorer succeeds',
                     'source baseline actual GPU worker exits', 'author source hashes unchanged'],
        registration_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        claim_boundary='Synchronization-corrected reproduction of the released shell recipe. This runtime execution differs from untouched author source; no arithmetic, alpha, beta, input or score change. Both source-as-is and corrected results are retained regardless of mean.')
    plan['parameters']['source_caller_stream_wait'] = True
    save(paths['plan'], plan)
    return plan, paths


def verify_launch_gate(plan):
    baseline = json.loads((RUN01 / 'state.json').read_text())
    if baseline['status'] != 'complete':
        return False, baseline['status']
    assert baseline['generation_exit_code'] == baseline['score_exit_code'] == 0
    assert baseline['n_predictions'] == 8910
    identities = json.loads((RUN01 / 'process_identity.json').read_text())['processes']
    previous_worker = next(x for x in identities if x['role'] == 'worker')
    proc = Path('/proc') / str(previous_worker['pid']) / 'stat'
    if proc.exists() and int(proc.read_text().split()[21]) == previous_worker['linux_start_ticks']:
        return False, 'old_worker_not_yet_exited'
    root = Path(plan['source_root'])
    assert all(sha(root / name) == value for name, value in plan['source_files'].items())
    return True, 'all gates passed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--wait-for-source-completion', action='store_true')
    args = parser.parse_args()
    plan, paths = prepare()
    if args.prepare_only:
        print(json.dumps({'prepared': True, 'run_id': plan['run_id'], 'paths': plan['paths']}))
        return
    if paths['predictions'].exists() and paths['predictions'].stat().st_size:
        raise RuntimeError('Refusing to overwrite previous control predictions')
    state = {'run_id': plan['run_id'], 'controller_pid': os.getpid(), 'registered_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
             'status': 'awaiting_source_baseline_completion', 'source_runtime_patches': plan['source_runtime_patches']}
    save(paths['state'], state)
    while True:
        passed, reason = verify_launch_gate(plan)
        if passed:
            break
        if reason in {'generation_failed', 'score_failed', 'completion_gate_failed'}:
            state.update(status='blocked_by_source_baseline_failure', reason=reason)
            save(paths['state'], state)
            raise RuntimeError(reason)
        if not args.wait_for_source_completion:
            raise RuntimeError('Source completion gate not yet passed: ' + reason)
        time.sleep(5)
    subprocess.run([os.sys.executable, str(OUT / 'summarize_completed_run.py')], check=True)
    env = os.environ.copy()
    for name in list(env):
        if any(name.startswith(prefix) for prefix in plan['cleared_environment_prefixes']):
            del env[name]
    env.update(plan['environment_overrides'])
    started = time.time()
    state.update(status='starting', started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), launch_gate=reason)
    save(paths['state'], state)
    with paths['generation_log'].open('w') as log:
        worker = subprocess.Popen(plan['generation_command'], cwd=plan['generation_cwd'], env=env,
                                  stdout=log, stderr=subprocess.STDOUT)
        stat_fields = (Path('/proc') / str(worker.pid) / 'stat').read_text().split()
        identity = {'controller_pid': os.getpid(), 'worker_pid': worker.pid,
                    'worker_start_ticks': int(stat_fields[21]), 'linux_session_id': int(stat_fields[5]),
                    'generation_command': plan['generation_command'], 'generation_cwd': plan['generation_cwd']}
        save(RUN02 / 'process_identity.json', identity)
        state.update(status='generating', worker_pid=worker.pid)
        save(paths['state'], state)
        code = worker.wait()
    state.update(generation_exit_code=code, generation_elapsed_seconds=time.time() - started)
    if code:
        state.update(status='generation_failed')
        save(paths['state'], state)
        raise SystemExit(code)
    rows = [json.loads(x) for x in paths['predictions'].read_text().splitlines()]
    questions = [json.loads(x) for x in (OUT / 'inputs/llava_pope_test.jsonl').read_text().splitlines()]
    assert len(rows) == 8910 and [r['question_id'] for r in rows] == [q['question_id'] for q in questions]
    state.update(status='scoring', n_predictions=len(rows), predictions_sha256=sha(paths['predictions']))
    save(paths['state'], state)
    with paths['score_log'].open('w') as log:
        scored = subprocess.run(plan['score_command'], cwd=plan['generation_cwd'], env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    state.update(score_exit_code=scored.returncode, status='complete' if scored.returncode == 0 else 'score_failed',
                 completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(), total_elapsed_seconds=time.time() - started)
    save(paths['state'], state)
    print(json.dumps(state, indent=2))
    raise SystemExit(scored.returncode)


if __name__ == '__main__':
    main()
