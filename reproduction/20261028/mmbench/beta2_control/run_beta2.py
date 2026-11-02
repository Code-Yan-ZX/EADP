#!/usr/bin/env python3
"""CPU preparation only by default; one future run requires explicit root slot allocation."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
CONTROL = BASE / 'streamwait_control'
PYTHON = '/home/dell/miniconda3/envs/llava_pruner/bin/python'
PRED = OUT / 'v15_EADP_K128_oldEN_beta2_alpha05_streamwait.jsonl'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())


def save(name, record):
    (OUT / name).write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_plan(plan):
    assert plan['parameters']['K'] == 128 and plan['parameters']['alpha'] == .5 and plan['parameters']['beta'] == 2
    assert plan['dataset']['n_rows'] == 4377 and plan['dataset']['n_circular_groups'] == 1176
    assert sha(plan['dataset']['path']) == plan['dataset']['sha256']
    for record in plan['source_files']:
        assert sha(record['path']) == record['sha256'], record['path']
    baseline = read(CONTROL / 'protocol.json')
    assert float(plan['command'][plan['command'].index('--beta') + 1]) == 2
    assert float(baseline['command'][baseline['command'].index('--beta') + 1]) == 1
    parameters = copy.deepcopy(plan['parameters'])
    parameters['beta'] = 1
    assert parameters == baseline['parameters']
    assert plan['dataset'] == baseline['dataset']
    old, new = copy.deepcopy(baseline['command']), copy.deepcopy(plan['command'])
    for command in (old, new):
        command[1:3] = ['ISOLATED_WRAPPER', 'ISOLATED_OUTPUT']
        command[command.index('--answers-file') + 1] = 'ISOLATED_PREDICTIONS'
        command[command.index('--beta') + 1] = 'EXPECTED_BETA_SINGLE_FACTOR'
    assert old == new
    patcher = load_module('frozen_beta_control_patch', CONTROL / 'patch_only.py')
    _, validation = patcher.build_patch()
    assert validation == plan['runtime_patch'] == baseline['runtime_patch']
    verifier = load_module('frozen_beta_control_verifier', OUT / 'verify_beta2.py')
    verifier.adapted_verifier()
    assert 'torch' not in sys.modules


def prepare():
    assert not PRED.exists() and not (OUT / 'state.json').exists() and not (OUT / 'protocol.json').exists()
    history = read(OUT / 'history_inventory.json')
    assert not history['matching_full_old_EN_v15_K128_alpha05_beta2_wait_found']
    baseline = read(CONTROL / 'protocol.json')
    verified = read(CONTROL / 'independent_verification.json')
    paired = read(CONTROL / 'paired_full_comparison.json')
    assert verified['success'] and verified['n_rows'] == 4377 and verified['n_circular_groups'] == 1176
    assert paired['success'] and paired['raw_answer_changed_rows'] == 0
    assert all(item['original_worker_exited'] for item in paired['worker_exit'])
    plan = copy.deepcopy(baseline)
    plan['command'] = [PYTHON, str(OUT / 'worker.py'), str(OUT / 'observed')] + baseline['command'][3:]
    plan['command'][plan['command'].index('--answers-file') + 1] = str(PRED)
    plan['command'][plan['command'].index('--beta') + 1] = '2.0'
    plan['parameters']['beta'] = 2
    plan['prepared_utc'] = datetime.now(timezone.utc).isoformat()
    plan['protocol'] = 'Single-factor beta1->2 full oldEN4377 K128 alpha0.5 with identical three-wait implementation'
    plan['baseline'] = {'protocol_path': str(CONTROL / 'protocol.json'), 'protocol_sha256': sha(CONTROL / 'protocol.json'),
                        'verification_path': str(CONTROL / 'independent_verification.json'),
                        'verification_sha256': sha(CONTROL / 'independent_verification.json'),
                        'prediction_sha256': verified['prediction_sha256'], 'beta': 1,
                        'circular_accuracy_percent': verified['circular_accuracy_percent']}
    plan['hypothesis_and_decision'] = {'purpose': 'Separate dataset-revision/beta confounding between historical V11 beta2 and new oldEN beta1.',
                                       'single_factor': 'Native --beta1.0 ->2.0 only; identical three waits in both arms.',
                                       'not_paper_argv_claim': True, 'score_threshold_gate': False,
                                       'followup_beta_sweep': False, 'all_outcomes_retained': True,
                                       'GPU_launch_status': 'Not allocated; CPU plan only; no GPU waiter or queue created.'}
    references = ['image_manifest.json', 'comparison_reference.json', 'cpu_dependency_input_equivalence.json',
                  'cpu_llm_dependency_equivalence.json', 'cpu_inputs_current.json', 'cpu_llm_tokens_current.json']
    for name in references:
        shutil.copyfile(BASE / name, OUT / name)
        assert sha(BASE / name) == sha(OUT / name)
    extra = [OUT / name for name in ('worker.py', 'run_beta2.py', 'verify_beta2.py', 'inventory_history.py', 'history_inventory.json')]
    extra += [CONTROL / name for name in ('protocol.json', 'state.json', 'independent_verification.json', 'paired_full_comparison.json')]
    extra += [OUT / name for name in references]
    plan['source_files'] += [{'path': str(path), 'sha256': sha(path)} for path in extra]
    plan['limits'] = ['Only beta changes from completed oldEN beta1/wait control; data/model/prompts/native decoding1024/scoring/3wait remain fixed.',
                      'This isolates a previously used beta2 setting; it does not search parameters or establish unpublished paper argv.',
                      'Keep historical V11 beta2 and both new beta1 runs; never select the highest score as a unified reproduction.',
                      'Concurrent GPU load and low-level scheduling cannot be replayed identically.',
                      'Local exact-letter circular metric retains all1176 groups; external server matching remains separate.']
    save('protocol.json', plan)
    check_plan(plan)
    save('READY.json', {'status': 'prepared_not_started', 'GPU_loaded': False, 'GPU_waiter_created': False,
                       'coordinator_allocation_required': True, 'protocol_sha256': sha(OUT / 'protocol.json'),
                       'CPU_gates_passed': ['Complete prior beta1 4377/1176 verified and exited', 'No exact beta2 run found in bounded inventory',
                                            'Only beta differs in normalized native argv and parameter map', 'Identical three-wait AST proof',
                                            'All frozen sources and full dataset SHA verified', 'Verifier AST differs only OUT and expected beta'],
                       'run_command': ['env', 'CUDA_VISIBLE_DEVICES=0', PYTHON, str(OUT / 'run_beta2.py'),
                                       'run', '--allocation', 'ROOT_EXPLICIT_FUTURE_SLOT_ALLOCATION']})
    print(json.dumps(read(OUT / 'READY.json'), ensure_ascii=False, indent=2))


def convert_submission():
    # Reuse the exact completed beta1 control's isolated author converter code.
    sys.path.insert(0, str(CONTROL))
    module = load_module('unchanged_beta1_converter', CONTROL / 'run_control.py')
    module.OUT, module.PRED = OUT, PRED
    module.convert_submission()


def run(allocation):
    assert allocation, 'Root must explicitly allocate a future GPU slot'
    plan = read(OUT / 'protocol.json')
    assert sha(OUT / 'protocol.json') == read(OUT / 'READY.json')['protocol_sha256']
    assert not PRED.exists() and not (OUT / 'state.json').exists(), 'Preserve all started runs'
    check_plan(plan)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True)
    assert len([line for line in apps.splitlines() if line.strip()]) < 2
    save('allocation.json', {'root_authorization': allocation, 'recorded_utc': datetime.now(timezone.utc).isoformat(),
                             'GPU_processes_before_launch': apps, 'protocol_sha256': sha(OUT / 'protocol.json')})
    module = load_module('unchanged_full_MMB_runner', BASE / 'run_author.py')
    module.OUT, module.PRED, module.DATA = OUT, PRED, Path(plan['dataset']['path'])
    module.command = lambda: plan['command']
    module.convert_submission = convert_submission
    module.run()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(BASE / 'vendor_scoring'))
    subprocess.run([PYTHON, str(OUT / 'verify_beta2.py')], env=env, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'check', 'run'))
    parser.add_argument('--allocation')
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'check':
        check_plan(read(OUT / 'protocol.json'))
        print(json.dumps({'CPU_gates_passed': True, 'GPU_loaded': False, 'would_launch': False}))
    else: run(args.allocation)
