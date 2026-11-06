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
D = B / 'e6/pixel_wavefront_model_t020'
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
    planfile = D / 'preflight_plan002.json'
    cpu = D / 'cpu_check004/verification.json'
    cm = D / 'monitor_cpu004'
    plan = read(planfile)
    c = read(cpu)
    verify_supervisor(cm, D / 'cpu_validation.py', cpu.parent)
    verify_supervisor(D / 'monitor_freeze003', D / 'freeze_preflight_plan.py', planfile)
    actual = {path: sha(path) for path in plan['source_sha256']}
    if (plan['source_sha256'] != actual or c['source_sha256_before'] != actual or
        c['source_sha256_after'] != actual or c['status'] != 'passed' or
        c['GT_opened'] is not False or c['gpu_used'] is not False or c['model_loaded'] is not False or c['existing_DEV_predictions_opened'] is not True):
        raise ValueError('frozen sources or CPU check mismatch')
    expected_tests = {'test_identity_contract.py', 'test_identity_worker.py',
                      'test_reference_contract.py', 'test_runtime_support.py', 'test_admission.py',
                      'test_single_arm.py', 'test_offline_identity.py', 'test_p3_reconstruct.py', 'test_schedule_contract.py'}
    if {x['script'] for x in c['tests']} != expected_tests or any(x['returncode'] != 0 for x in c['tests']):
        raise ValueError('missing CPU tests')
    if (plan['stage'] != 'three-anchor wavefront model no-GPU preflight only' or
        plan['threshold'] != 0.2 or plan['GT_opened'] is not False or
        plan['existing_DEV_predictions_opened'] is not True or plan['scoring_performed'] is not False or
        plan['gpu_authorized'] is not False or len(plan['images']) != 3 or
        {x['anchor']['task'] for x in plan['images']} != {'DocVQA_VAL', 'TextVQA_VAL', 'AI2D_TEST'}):
        raise ValueError('incorrect preflight scope')
    if (plan['cpu_sha256'] != sha(cpu) or plan['cpu_started_sha256'] != sha(cm/'started.json') or
        plan['cpu_terminal_sha256'] != sha(cm/'terminal.json') or
        plan['component_review_sha256'] != sha(R/'pixel_wavefront_t020_gpu32_result_review.json') or
        plan['budget_source_sha256'] != sha(B/'initial_plan.json')):
        raise ValueError('evidence hash mismatch')
    component=read(R/'pixel_wavefront_t020_gpu32_result_review.json')
    complete=B/'e6/pixel_wavefront_t020_gpu32/run_smoke001/observations/complete.json'
    if component.get('status')!='passed' or component.get('images')!=32 or component.get('ordered_equal')!=32 or component.get('evidence_sha256',{}).get(str(complete))!=sha(complete):
        raise ValueError('independent component proof mismatch')
    cases={x['key']:x for x in read(complete)['cases']}
    for item in plan['images']:
        a=item['anchor'];key=a['task']+'/'+str(a['qid']);case=cases[key]
        obs=B/'e6/pixelprune_t020_smoke/run_smoke001/observations'/(a['task']+'_'+str(a['qid'])+'.json')
        smoke=read(obs);expected=item['expected']['selected_indices']
        if sha(obs)!=item['observation_sha256'] or smoke['threshold']!=0.2 or smoke['anchored'] is not True or smoke['selected_indices']!=expected or case['ordered_equal'] is not True or len(case['arms'])!=2 or any(arm['indices']!=expected for arm in case['arms']):
            raise ValueError('model anchor / GPU component ordered index mismatch')
    aliases = plan.get('model_metadata_aliases_sha256', {})
    if len(aliases) != 7 or any(actual.get(path) != digest or sha(path) != digest or not Path(path).is_symlink() for path, digest in aliases.items()):
        raise ValueError('model metadata alias or content mismatch')
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
             'GT_opened': False, 'existing_DEV_predictions_opened': True, 'gpu_authorized': False, 'reviewer_sha256': sha(__file__),
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
