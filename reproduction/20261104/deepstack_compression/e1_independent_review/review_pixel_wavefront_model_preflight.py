"""Independent review: admit only the real no-GPU wavefront model preflight."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
B = R.parent
D = B / 'e6/pixel_wavefront_model'
ROOT = B.parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_supervisor(directory, script, out):
    start = read(directory / 'started.json')
    end = read(directory / 'terminal.json')
    cmd = start['command']
    cwd = Path(start['cwd'])
    if (end['reason'] != 'completed' or end['worker_returncode'] != 0 or
        end['remaining_process_group'] != [] or start['worker_pid'] != end['worker_pid'] or
        (cwd / cmd[1]).resolve() != script.resolve() or cmd.count('--out') != 1 or
        (cwd / cmd[cmd.index('--out') + 1]).resolve() != out.resolve()):
        raise ValueError('supervisor evidence mismatch')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.resolve() != D / 'independent_preflight_review.json':
        raise ValueError('unexpected output')
    planfile = D / 'preflight_plan001.json'
    cpu = D / 'cpu_check002/verification.json'
    cm = D / 'monitor_cpu002'
    plan = read(planfile)
    c = read(cpu)
    verify_supervisor(cm, D / 'cpu_validation.py', cpu.parent)
    verify_supervisor(D / 'monitor_freeze001', D / 'freeze_preflight_plan.py', planfile)
    actual = {path: sha(path) for path in plan['source_sha256']}
    if (plan['source_sha256'] != actual or c['source_sha256_before'] != actual or
        c['source_sha256_after'] != actual or c['status'] != 'passed' or
        c['GT_opened'] is not False or c['gpu_used'] is not False or c['model_loaded'] is not False):
        raise ValueError('frozen sources or CPU check mismatch')
    expected_tests = {'test_identity_contract.py', 'test_identity_worker.py',
                      'test_reference_contract.py', 'test_runtime_support.py', 'test_admission.py'}
    if {x['script'] for x in c['tests']} != expected_tests or any(x['returncode'] != 0 for x in c['tests']):
        raise ValueError('missing CPU tests')
    if (plan['stage'] != 'three-anchor wavefront model no-GPU preflight only' or
        plan['threshold'] != 0.45 or plan['GT_opened'] is not False or
        plan['existing_DEV_predictions_opened'] is not True or plan['scoring_performed'] is not False or
        plan['gpu_authorized'] is not False or len(plan['images']) != 3 or
        {x['anchor']['task'] for x in plan['images']} != {'DocVQA_VAL', 'TextVQA_VAL', 'AI2D_TEST'}):
        raise ValueError('incorrect preflight scope')
    if (plan['cpu_sha256'] != sha(cpu) or plan['cpu_started_sha256'] != sha(cm/'started.json') or
        plan['cpu_terminal_sha256'] != sha(cm/'terminal.json') or
        plan['component_review_sha256'] != sha(R/'pixel_wavefront_gpu32_v2_result_review.json') or
        plan['budget_source_sha256'] != sha(B/'initial_plan.json')):
        raise ValueError('evidence hash mismatch')
    required = [D/'preflight_entry.py', D/'preflight_launch.py', D/'runtime_support.py',
                D/'admission.py', D/'source_inventory.py', D/'identity_worker.py',
                B/'e6/pixel_wavefront_cpu/candidate.py', B/'monitor/supervise.py']
    if any(actual.get(str(x.resolve())) != sha(x) for x in required):
        raise ValueError('required input unbound')
    sys.path.insert(0, str(D))
    import admission
    try:
        admission.require_gpu()
    except ValueError:
        pass
    else:
        raise ValueError('GPU gate must remain locked')
    source = ROOT/'reproduction/20261028/qwen/gpu_formula_facility_panel_run02/run_panel.py'
    spec = importlib.util.spec_from_file_location('wavefront_independent_container_review', source)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    author_plan = read(source.parent/'run_plan.json')
    command = runtime.container_command(author_plan, [str(D/'preflight_entry.py'), '--out',
                                         str(D/'run_preflight001/worker_preflight.json')], False, 'eadp-review-cpu')
    if ('--gpus' in command or '--device' in command or
        'CUDA_VISIBLE_DEVICES=' not in command or command.count('--rm') != 1):
        raise ValueError('constructed no-GPU container is not isolated as planned')
    proof = {'status': 'passed', 'independent_review': True, 'preflight_only': True,
             'GT_opened': False, 'gpu_authorized': False, 'reviewer_sha256': sha(__file__),
             'source_sha256': actual, 'plan_sha256': sha(planfile), 'cpu_sha256': sha(cpu),
             'cpu_started_sha256': sha(cm/'started.json'), 'cpu_terminal_sha256': sha(cm/'terminal.json'),
             'errors': [], 'limitations': ['This review admits only a no-GPU processor/weight preflight.',
                        'The real preflight, identity controller and later GPU/timing review remain required.',
                        'Prior DEV predictions were used for same-arm identity.']}
    with args.out.open('x') as f:
        json.dump(proof, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


if __name__ == '__main__':
    main()
