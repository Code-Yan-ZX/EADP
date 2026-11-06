"""Fail-closed future GPU admission for the paired timing experiment.

Requires the actual completed three-anchor acceptance, this execution layer's
future same-image no-device acceptance, frozen analysis/formal plans, and an
independent GPU prelaunch review. Missing evidence rejects before GPU queries.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REV = ROOT / 'e1_independent_review'
IDENTITY = ROOT / 'e6/rres_eadp_identity3_execution'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def ended(start, terminal):
    a, b = read(start), read(terminal)
    return type(a.get('worker_pid')) is int and a['worker_pid'] > 0 and b.get('worker_pid') == a['worker_pid'] and b.get('reason') == 'completed' and b.get('worker_returncode') == 0 and b.get('remaining_process_group') == []


def require_identity():
    proof = REV / 'rres_eadp_identity3_gpu_result_review.json'
    q = read(proof)
    monitor = REV / 'monitor_rres_eadp_identity3_gpu_result'
    files = {'formal_sha256': IDENTITY/'verification001.json', 'formal_started_sha256': IDENTITY/'monitor_verify001/started.json',
             'formal_terminal_sha256': IDENTITY/'monitor_verify001/terminal.json', 'run_started_sha256': IDENTITY/'monitor_run001/started.json',
             'run_terminal_sha256': IDENTITY/'monitor_run001/terminal.json', 'source_manifest_sha256': IDENTITY/'run001_sources/manifest.json',
             'reviewer_sha256': REV/'review_rres_eadp_identity3_gpu_result.py'}
    if q.get('status') != 'passed' or q.get('independent_review') is not True or q.get('images') != 3 or q.get('generations') != 6 or q.get('GT_opened') is not False or q.get('scoring_performed') is not False or q.get('timing_claims') is not False or q.get('errors') != []:
        raise ValueError('three-anchor identity result')
    if any(q.get(k) != sha(v) for k, v in files.items()) or not ended(monitor/'started.json', monitor/'terminal.json') or not ended(files['formal_started_sha256'], files['formal_terminal_sha256']) or not ended(files['run_started_sha256'], files['run_terminal_sha256']):
        raise ValueError('three-anchor identity evidence')
    if len(q.get('files_sha256', {})) != 12 or any(sha(path) != digest for path, digest in q['files_sha256'].items()):
        raise ValueError('three-anchor saved files')
    return sha(proof)


def require_execution_preflight():
    proof = REV / 'rres_eadp_timing_execution_preflight_result_review.json'
    reviewer = REV / 'review_rres_eadp_timing_execution_preflight_result.py'
    monitor = REV / 'monitor_rres_eadp_timing_execution_preflight_result'
    run = HERE / 'cpu_preflight001'
    own_monitor = HERE / 'monitor_cpu_preflight001'
    q = read(proof)
    files = {'plan_sha256': HERE/'cpu_plan001.json', 'worker_sha256': run/'worker_preflight.json',
             'launch_sha256': run/'launch.json', 'identity_sha256': run/'container_identity.json',
             'exit_sha256': run/'exit.json', 'started_sha256': own_monitor/'started.json',
             'terminal_sha256': own_monitor/'terminal.json', 'reviewer_sha256': reviewer}
    if q.get('status') != 'passed' or q.get('independent_review') is not True or q.get('preflight_only') is not True or q.get('GPU_authorized') is not False or q.get('GT_opened') is not False or q.get('existing_predictions_opened') is not True or q.get('errors') != []:
        raise ValueError('execution noGPU result')
    if any(q.get(k) != sha(v) for k, v in files.items()) or not ended(monitor/'started.json', monitor/'terminal.json') or not ended(own_monitor/'started.json', own_monitor/'terminal.json'):
        raise ValueError('execution noGPU evidence')
    w, e = read(files['worker_sha256']), read(files['exit_sha256'])
    if w.get('status') != 'passed' or w.get('cuda_initialized') is not False or w.get('model_loaded') is not False or e.get('container_exit_code') != 0 or e.get('container_stopped_verified') is not True:
        raise ValueError('execution noGPU worker/container')
    return sha(proof)


def admit():
    plan = read(HERE/'gpu_plan001.json')
    if plan.get('scope') != 'rres_eadp_paired_timing_gpu' or any(sha(path) != digest for path, digest in plan['source_sha256'].items()):
        raise ValueError('GPU plan or source drift')
    identity = require_identity()
    if identity != plan.get('identity_review_sha256'):
        raise ValueError('three-anchor admission binding')
    if require_execution_preflight() != plan.get('execution_preflight_review_sha256'):
        raise ValueError('execution noGPU admission binding')
    formal = HERE/'verification_plan001.json'
    if read(formal).get('gpu_plan_sha256') != sha(HERE/'gpu_plan001.json'):
        raise ValueError('formal plan binding')
    proof = REV/'rres_eadp_timing_gpu_prelaunch_review.json'
    q = read(proof)
    reviewer = REV/'review_rres_eadp_timing_gpu_prelaunch.py'
    monitor = REV/'monitor_rres_eadp_timing_gpu_prelaunch'
    if q.get('status') != 'passed' or q.get('independent_review') is not True or q.get('scope') != 'rres_eadp_paired_timing_gpu_prelaunch' or q.get('errors') != [] or q.get('GT_opened') is not False or q.get('plan_sha256') != sha(HERE/'gpu_plan001.json') or q.get('verification_plan_sha256') != sha(formal) or q.get('reviewer_sha256') != sha(reviewer) or q.get('source_sha256') != plan['source_sha256'] or not ended(monitor/'started.json', monitor/'terminal.json'):
        raise ValueError('independent GPU prelaunch review')
    return {'plan_sha256': sha(HERE/'gpu_plan001.json'), 'identity_review_sha256': identity, 'review_sha256': sha(proof)}
