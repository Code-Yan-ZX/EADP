#!/usr/bin/env python3
"""CPU-only final completion/exit proof for the next owner's slot-B handoff."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'run02_source_wait_only_k32'


def save(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    fields = (Path('/proc') / str(os.getpid()) / 'stat').read_text().split()
    save(RUN / 'finalizer_identity.json', {'pid': os.getpid(), 'start_ticks': int(fields[21]),
        'linux_session': int(fields[5]), 'type': 'CPU-only current-job completion monitor; no CUDA imports or model loading'})
    while True:
        state = json.loads((RUN / 'state.json').read_text())
        if state['status'] == 'complete':
            break
        if state['status'] in {'generation_failed', 'score_failed', 'completion_gate_failed', 'blocked_by_source_baseline_failure'}:
            raise RuntimeError('Cannot finalize failed control: ' + state['status'])
        time.sleep(5)
    assert state['n_predictions'] == 8910 and state['generation_exit_code'] == state['score_exit_code'] == 0
    identity = json.loads((RUN / 'process_identity.json').read_text())
    proc = Path('/proc') / str(identity['worker_pid']) / 'stat'
    old_worker_alive = proc.exists() and int(proc.read_text().split()[21]) == identity['worker_start_ticks']
    assert not old_worker_alive
    plan = json.loads((RUN / 'run_plan.json').read_text())
    source_checks = {name: digest(Path(plan['source_root']) / name) == expected for name, expected in plan['source_files'].items()}
    assert all(source_checks.values())
    subprocess.run([sys.executable, str(ROOT / 'compare_completed_controls.py')], check=True)
    summary = json.loads((RUN / 'independently_verified_full_summary.json').read_text())
    independent = sum(c['f1_percent'] for c in summary['categories'].values()) / 3
    assert abs(independent - summary['official_macro_f1_percent']) < 1e-12
    assert summary['predictions']['sha256'] == state['predictions_sha256']
    summary.update(independent_macro_f1_percent=independent, predictions_sha256=summary['predictions']['sha256'],
        official_scorer_exit_code=0, all_8910_question_ids_order_verified=True,
        source_sha_unchanged_checks=source_checks, gpu_worker_exit_verified=True,
        verification_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    save(RUN / 'full_result_summary.json', summary)
    release = {'observed_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'GPU_slot': 'B',
        'next_owner': '/root/v2_review', 'control_run': RUN.name, 'status': 'all_completion_and_exit_gates_passed',
        'worker_pid': identity['worker_pid'], 'worker_start_ticks': identity['worker_start_ticks'],
        'same_worker_still_alive': old_worker_alive, 'generation_exit_code': 0, 'official_score_exit_code': 0,
        'prediction_count': 8910, 'predictions_sha256': state['predictions_sha256'],
        'independent_and_official_macro_match': True, 'all_source_sha_match': True,
        'full_result_summary': {'path': str(RUN / 'full_result_summary.json'), 'sha256': digest(RUN / 'full_result_summary.json')},
        'no_more_POPE_models_scheduled': True, 'note': 'This is completion and worker-exit evidence, independent of accuracy value or consistency result. The next owner must perform its own launch guards.'}
    save(RUN / 'completion_and_release_proof.json', release)
    print(json.dumps(release, indent=2), flush=True)


if __name__ == '__main__':
    main()
