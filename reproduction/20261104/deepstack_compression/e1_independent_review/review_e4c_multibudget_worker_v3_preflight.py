"""Independent review of 32-image timing worker no-GPU preflight admission only."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

R = Path(__file__).resolve().parent
B = R.parent
D = B/'e4_c_multibudget_worker_v3'
ROOT = B.parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out.resolve()
    if out != R/'e4c_multibudget_worker_v3_preflight_review.json':
        raise ValueError('unexpected proof output')
    plan_path, cpu_path = D/'runtime_plan001.json', D/'execution_cpu001.json'
    start, end = D/'monitor_execution_cpu001/started.json', D/'monitor_execution_cpu001/terminal.json'
    p, c, s, t = map(read, (plan_path, cpu_path, start, end))
    if (t.get('reason') != 'completed' or t.get('worker_returncode') != 0 or
        t.get('remaining_process_group') != [] or s.get('worker_pid') != t.get('worker_pid') or
        (Path(s['cwd'])/s['command'][1]).resolve() != D/'freeze_execution.py'):
        raise ValueError('CPU freeze supervisor mismatch')
    sources = {path: sha(path) for path in p['source_sha256']}
    if (len(sources) != 245 or sources != p['source_sha256'] or sources != c['source_sha256'] or
        c.get('status') != 'passed' or c.get('test_returncode') != 0 or 'Ran 21 tests' not in c.get('test_output', '') or
        c.get('plan_sha256') != sha(plan_path) or c.get('GT_opened') is not False or
        c.get('predictions_opened') is not True or c.get('scoring_performed') is not False or
        c.get('model_loaded') is not False or c.get('GPU_authorized') is not False):
        raise ValueError('CPU or source evidence mismatch')
    if (p.get('stage') != 'post-hoc 32-image 4-budget natural/fixed32 timing pilot' or p.get('jobs') != 1520 or
        p.get('GT_opened') is not False or p.get('predictions_opened') is not True or
        p.get('scoring_performed') is not False or p.get('GPU_authorized_by_plan') is not False or
        (p.get('inner_timeout_seconds'), p.get('outer_timeout_seconds'), p.get('stale_seconds'),
         p.get('grace_seconds'), p.get('acceptance_reserve_seconds')) != (7200, 7320, 1800, 90, 3600) or
        p.get('budget_source_sha256') != sha(B/'initial_plan.json') or
        p.get('pilot_plan_sha256') != sha(B/'e4_c_multibudget_expansion/pilot_plan001.json')):
        raise ValueError('plan contract mismatch')
    for name in ('admission.py','launch.py','worker.py','worker_core.py','runner.py','fixed_reference_gate.py','test_execution.py'):
        if sources.get(str(D/name)) != sha(D/name):
            raise ValueError('unbound runtime source')
    gate_spec = importlib.util.spec_from_file_location('timing_worker_review_admission', D/'admission.py')
    gate = importlib.util.module_from_spec(gate_spec)
    gate_spec.loader.exec_module(gate)
    try:
        gate.admit_gpu()
    except ValueError as exc:
        if 'locked' not in str(exc).lower():
            raise
    else:
        raise ValueError('timing GPU gate unexpectedly open')
    if (R/'e4c_multibudget_worker_v3_prelaunch_review.json').exists():
        raise ValueError('unexpected existing GPU proof')
    source = ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_panel.py'
    spec = importlib.util.spec_from_file_location('bulk_reference_review_runtime', source)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cmd = m.container_command(read(source.parent/'run_plan.json'),
        [str(D/'worker.py'), '--out', str(D/'run_preflight001'), '--preflight-only'], False, 'bulk-review')
    if '--gpus' in cmd or '--device' in cmd or 'CUDA_VISIBLE_DEVICES=' not in cmd or cmd.count('--rm') != 1:
        raise ValueError('unexpected GPU exposure in CPU container')
    proof = {'status': 'passed', 'independent_review': True, 'scope': 'multibudget_worker_v3_cpu_preflight_only',
             'GT_opened': False, 'predictions_opened': True, 'scoring_performed': False,
             'source_sha256': sources, 'plan_sha256': sha(plan_path), 'execution_cpu_sha256': sha(cpu_path),
             'execution_cpu_started_sha256': sha(start), 'execution_cpu_terminal_sha256': sha(end),
             'reviewer_sha256': sha(__file__), 'errors': [], 'limitations': [
                 'Only same-image no-GPU preflight admitted; 248 fixed32 references and final timing prelaunch remain separately gated.',
                 'Previously accepted DEV prediction sources opened for SHA hashing; no ground truth read.',
                 'A complete reference panel and independent post-run acceptance are required before timing.']}
    with out.open('x') as f:
        json.dump(proof, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


if __name__ == '__main__':
    main()
