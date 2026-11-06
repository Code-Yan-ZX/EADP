"""Review the fixed32 CPU-only container preflight admission, not the GPU run."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

R = Path(__file__).resolve().parent
B = R.parent
D = B / 'e4_c_multibudget_fixed32'
ROOT = B.parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    out = ap.parse_args().out.resolve()
    if out != R / 'e4c_multibudget_fixed32_worker_preflight_review.json':
        raise ValueError('unexpected review output')
    plan_path, cpu_path = D / 'execution_plan005.json', D / 'execution_cpu005.json'
    start, end = D / 'monitor_execution_cpu005/started.json', D / 'monitor_execution_cpu005/terminal.json'
    p, c, s, t = map(read, (plan_path, cpu_path, start, end))
    command = s['command']
    if (t.get('reason') != 'completed' or t.get('worker_returncode') != 0 or
        t.get('remaining_process_group') != [] or s.get('worker_pid') != t.get('worker_pid') or
        (Path(s['cwd']) / command[1]).resolve() != D / 'freeze_execution.py'):
        raise ValueError('CPU freeze supervisor mismatch')
    actual = {path: sha(path) for path in p['source_sha256']}
    if (len(actual) != 164 or actual != p['source_sha256'] or actual != c['source_sha256'] or
        c.get('status') != 'passed' or c.get('test_returncode') != 0 or
        'Ran 19 tests' not in c.get('test_output', '') or c.get('plan_sha256') != sha(plan_path) or
        c.get('GT_opened') is not False or c.get('predictions_opened') is not True or
        c.get('scoring_performed') is not False or c.get('model_loaded') is not False or
        c.get('GPU_authorized') is not False):
        raise ValueError('CPU tests or frozen source mismatch')
    if (p.get('stage') != '4-budget fixed32 24-generation diagnostic' or
        p.get('budgets') != [128, 256, 512, 1024] or p.get('GT_opened') is not False or
        p.get('predictions_opened') is not True or p.get('scoring_performed') is not False or
        p.get('GPU_authorized_by_plan') is not False or
        (p.get('inner_timeout_seconds'), p.get('outer_timeout_seconds'), p.get('stale_seconds'),
         p.get('grace_seconds'), p.get('acceptance_reserve_seconds')) != (3600, 3720, 1200, 90, 3600) or
        p.get('budget_source_sha256') != sha(B / 'initial_plan.json')):
        raise ValueError('fixed32 plan contract')
    for name in ('gate.py', 'launch.py', 'worker.py', 'core.py', 'fresh_fixed.py',
                 'light_fixed.py', 'fixed_common.py', 'compare_fixed.py', 'test_execution.py'):
        if actual.get(str(D / name)) != sha(D / name):
            raise ValueError('unbound runtime source')
    if (R / 'e4c_multibudget_fixed32_prelaunch_review.json').exists():
        raise ValueError('unexpected existing GPU proof; review preflight only')
    runtime_path = ROOT / 'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_panel.py'
    spec = importlib.util.spec_from_file_location('fixed32_review_runtime', runtime_path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    cmd = runtime.container_command(read(runtime_path.parent / 'run_plan.json'),
        [str(D / 'worker.py'), '--out', str(D / 'run_preflight001'), '--preflight-only'], False, 'fixed32-review')
    if '--gpus' in cmd or '--device' in cmd or 'CUDA_VISIBLE_DEVICES=' not in cmd or cmd.count('--rm') != 1:
        raise ValueError('CPU container would expose GPU')
    proof = {'status': 'passed', 'independent_review': True, 'scope': 'fixed32_cpu_preflight_only',
             'GT_opened': False, 'predictions_opened': True, 'scoring_performed': False,
             'prediction_scope': 'previously accepted DEV prediction artifacts hashed for source identity',
             'source_sha256': actual, 'plan_sha256': sha(plan_path), 'execution_cpu_sha256': sha(cpu_path),
             'execution_cpu_started_sha256': sha(start), 'execution_cpu_terminal_sha256': sha(end),
             'reviewer_sha256': sha(__file__), 'errors': [], 'limitations': [
                 'Only the real same-image no-GPU preflight is admitted; no model or GPU identity established.',
                 'Full KV arrays are not archived; this diagnostic does not establish performance.',
                 'Existing DEV predictions were opened only for source hashing in this review.']}
    with out.open('x') as f:
        json.dump(proof, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


if __name__ == '__main__':
    main()
