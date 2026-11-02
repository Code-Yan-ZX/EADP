#!/usr/bin/env python3
"""Prepare or explicitly launch one single-alpha control; no scheduler or queue."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from verify_complete_predictions import verify as verify_predictions
from wait_only_patch import build

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'run03_single_alpha05_beta1_wait_k32'
BASE = ROOT / 'run02_source_wait_only_k32'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2) + '\n')
    temporary.replace(path)


def stat_guard(path):
    stat = path.stat()
    return {'resolved': str(path.resolve()), 'bytes': stat.st_size, 'device': stat.st_dev,
        'inode': stat.st_ino, 'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns}


def prepare():
    RUN.mkdir(exist_ok=True)
    if (RUN / 'predictions.jsonl').exists() or (RUN / 'process_identity.json').exists():
        raise RuntimeError('Refusing to replace a launched control registration')
    base = json.loads((BASE / 'run_plan.json').read_text())
    survey = json.loads((ROOT / 'historical_alpha_control_survey.json').read_text())
    assert not survey['exact_v15_K32_alpha05_beta1_nonanchor_metadata']
    plan = copy.deepcopy(base)
    command = base['generation_command'][:]
    command[1] = str(ROOT / 'worker_alpha_control.py')
    command[2:2] = ['--observer-output', str(RUN)]
    command[command.index('--alpha') + 1] = '0.5'
    command[command.index('--answers-file') + 1] = str(RUN / 'predictions.jsonl')
    plan.update(run_id=RUN.name, generation_command=command,
        parameters={**base['parameters'], 'alpha': .5},
        paths={key: str(RUN / Path(value).name) for key, value in base['paths'].items()},
        recipe_origin='Single-alpha control of complete wait-only1/1 run; .5/1 is also native generic-loader and README example default.',
        claim_boundary='Only alpha changes1->.5. This isolates finite global/dense fusion contribution under beta1; does not certify paper argv. Keep every complete outcome; no follow-up parameter tuning by score.',
        base_plan={'path': str(BASE / 'run_plan.json'), 'sha256': sha(BASE / 'run_plan.json')},
        base_summary={'path': str(BASE / 'full_result_summary.json'), 'sha256': sha(BASE / 'full_result_summary.json')},
        historical_search={'path': str(ROOT / 'historical_alpha_control_survey.json'), 'sha256': sha(ROOT / 'historical_alpha_control_survey.json')},
        registration_utc=datetime.now(timezone.utc).isoformat(),
        launch_policy='Prepared only; no queue or auto launch. Manual run requires --gpu-authorized and explicit root allocation note after NeXTfull.',
        observation_policy='Read-only post-native-load model/source identity only; native generation loop and generate remain unwrapped.')
    for key in ['trigger', 'launch_gate']:
        plan.pop(key, None)
    plan['score_command'] = base['score_command'][:]
    plan['score_command'][plan['score_command'].index('--result-file') + 1] = str(RUN / 'predictions.jsonl')
    _, proof = build()
    plan['wait_patch_prepared_proof'] = proof
    environment_reference = ROOT / 'run01_source_as_is_k32/runtime_environment.json'
    previous_environment = json.loads(environment_reference.read_text())
    plan['runtime_versions'] = {key: previous_environment[key] for key in ['torch', 'transformers']}
    plan['runtime_environment_reference'] = {'path': str(environment_reference), 'sha256': sha(environment_reference)}
    plan['expected_attention_backend'] = 'sdpa'
    historical = json.loads((ROOT / 'model_identity/model_identity_audit.json').read_text())
    plan['prior_model_payload_identity'] = {'path': str(ROOT / 'model_identity/model_identity_audit.json'),
        'sha256': sha(ROOT / 'model_identity/model_identity_audit.json'),
        'reuse_policy': 'Reuse completed public-LFS-matched SHA audit; current stat guards freeze unchanged payloads across this control, without rehashing13GB.'}
    plan['model_stat_guards'] = {}
    for model in historical['models'].values():
        for row in model['files']:
            path = Path(row['local_path'])
            assert row['byte_identity_match'] and path.stat().st_size == row['bytes']
            plan['model_stat_guards'][str(path)] = {**stat_guard(path), 'prior_verified_sha256': row['local_sha256']}
    images = json.loads((ROOT / 'canonical_image_manifest.json').read_text())
    plan['image_stat_guards'] = {row['path']: {**stat_guard(Path(row['path'])), 'prior_verified_sha256': row['sha256']} for row in images}
    plan['small_file_guards'] = {str(ROOT / name): sha(ROOT / name) for name in [
        'inputs/llava_pope_test.jsonl', 'pinned_pope_labels/coco_pope_random.json',
        'pinned_pope_labels/coco_pope_popular.json', 'pinned_pope_labels/coco_pope_adversarial.json',
        'canonical_image_manifest.json', 'model_identity/model_identity_audit.json']}
    plan['implementation_sha256'] = {str(ROOT / name): sha(ROOT / name) for name in [
        'alpha_control.py', 'worker_alpha_control.py', 'wait_only_patch.py', 'verify_complete_predictions.py']}
    save(RUN / 'run_plan.json', plan)
    save(RUN / 'state.json', {'run_id': RUN.name, 'status': 'prepared_not_scheduled', 'registration_utc': plan['registration_utc'], 'gpu_model_loaded': False})
    validate_frozen(plan)
    return plan


def validate_frozen(plan):
    source = Path(plan['source_root'])
    assert all(sha(source / name) == expected for name, expected in plan['source_files'].items())
    for group in ['small_file_guards', 'implementation_sha256']:
        assert all(sha(Path(name)) == expected for name, expected in plan[group].items()), group
    for group in ['model_stat_guards', 'image_stat_guards']:
        for path, guard in plan[group].items():
            assert stat_guard(Path(path)) == {k: v for k, v in guard.items() if k != 'prior_verified_sha256'}, path
    assert sha(Path(plan['base_summary']['path'])) == plan['base_summary']['sha256']
    baseline = json.loads(Path(plan['base_summary']['path']).read_text())
    assert baseline['n_predictions'] == 8910 and baseline['gpu_worker_exit_verified']


def alive(identity):
    proc = Path('/proc') / str(identity['worker_pid']) / 'stat'
    return proc.exists() and int(proc.read_text().split()[21]) == identity['worker_start_ticks']


def verify():
    plan = json.loads((RUN / 'run_plan.json').read_text())
    validate_frozen(plan)
    state = json.loads((RUN / 'state.json').read_text())
    assert state['generation_exit_code'] == 0
    identity = json.loads((RUN / 'process_identity.json').read_text())
    assert sha(RUN / 'run_plan.json') == identity['plan_sha256'] == state['plan_sha256']
    assert not alive(identity), 'actual worker must exit'
    actual = json.loads((RUN / 'actual_loaded_runtime.json').read_text())
    assert actual['worker_pid'] == identity['worker_pid']
    assert (actual['actual_alpha'], actual['actual_beta'], actual['actual_visual_token_num']) == (.5, 1., 32)
    assert actual['actual_load_args'][0] == plan['generation_command'][plan['generation_command'].index('--model-path') + 1]
    assert actual['native_argv'] == [plan['generation_command'][1]] + plan['generation_command'][4:]
    assert all(actual[key] == value for key, value in plan['runtime_versions'].items())
    assert actual['attention_backend'] == plan['expected_attention_backend']
    source = Path(plan['source_root'])
    for module, path in actual['loaded_module_paths'].items():
        relative = Path(path).relative_to(source)
        assert actual['loaded_source_sha256'][module] == plan['source_files'][str(relative)]
    patch = json.loads((RUN / 'runtime_patch_proof.json').read_text())
    expected = plan['wait_patch_prepared_proof']
    assert patch['runtime_patch_installed'] and patch['restoration_ast_identical']
    for key in ['production_source_sha256', 'original_forward_ast_sha256', 'patched_forward_ast_sha256', 'inserted_nodes']:
        assert patch[key] == expected[key], key
    result = verify_predictions(RUN / 'predictions.jsonl', RUN)
    assert result['model_ids'] == {'llava-v1.5-7b': 8910}
    result.update(gpu_worker_exit_verified=True, frozen_source_inputs_payload_stats_verified=True,
        actual_runtime_verified=True, run_plan_sha256=sha(RUN / 'run_plan.json'),
        actual_alpha=.5, actual_beta=1., actual_visual_token_num=32,
        comparison_reference=plan['base_summary'],
        alpha_control_minus_alpha1_F1=result['official_macro_f1_percent'] - json.loads((BASE / 'full_result_summary.json').read_text())['official_macro_f1_percent'],
        paper_recipe_identity='unknown; this control does not certify paperargv',
        acceptance_policy='All8910 identities/source/runtime/score must pass. Score direction and consistency outcome are reported, never used to discard the result.')
    save(RUN / 'full_result_summary.json', result)
    return result


def run(allocation_note):
    plan = json.loads((RUN / 'run_plan.json').read_text())
    validate_frozen(plan)
    if (RUN / 'predictions.jsonl').exists() or (RUN / 'process_identity.json').exists():
        raise RuntimeError('No overwrite/restart of existing control')
    env = os.environ.copy()
    for name in list(env):
        if any(name.startswith(prefix) for prefix in plan['cleared_environment_prefixes']):
            del env[name]
    env.update(plan['environment_overrides'])
    state = {'run_id': RUN.name, 'status': 'starting', 'allocation_note': allocation_note,
        'started_utc': datetime.now(timezone.utc).isoformat(), 'controller_pid': os.getpid(),
        'plan_sha256': sha(RUN / 'run_plan.json')}
    save(RUN / 'state.json', state)
    start = time.time()
    with (RUN / 'generation.log').open('w') as log:
        worker = subprocess.Popen(plan['generation_command'], cwd=plan['generation_cwd'], env=env,
            stdout=log, stderr=subprocess.STDOUT)
        fields = (Path('/proc') / str(worker.pid) / 'stat').read_text().split()
        identity = {'controller_pid': os.getpid(), 'worker_pid': worker.pid, 'worker_start_ticks': int(fields[21]),
            'linux_session_id': int(fields[5]), 'generation_command': plan['generation_command'],
            'started_utc': state['started_utc'], 'plan_sha256': state['plan_sha256']}
        save(RUN / 'process_identity.json', identity)
        state.update(status='generating', worker_pid=worker.pid)
        save(RUN / 'state.json', state)
        code = worker.wait()
    state.update(generation_exit_code=code, generation_elapsed_seconds=time.time() - start,
        status='validating' if code == 0 else 'generation_failed')
    save(RUN / 'state.json', state)
    if code:
        raise SystemExit(code)
    try:
        result = verify()
    except Exception:
        state.update(status='verification_failed')
        save(RUN / 'state.json', state)
        raise
    state.update(status='complete', n_predictions=8910, score_exit_code=0,
        official_macro_f1_percent=result['official_macro_f1_percent'],
        completed_utc=datetime.now(timezone.utc).isoformat())
    save(RUN / 'state.json', state)
    print(json.dumps(state))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'verify', 'run'])
    parser.add_argument('--gpu-authorized', action='store_true')
    parser.add_argument('--allocation-note')
    args = parser.parse_args()
    if args.command == 'prepare':
        plan = prepare()
        print(json.dumps({'prepared': True, 'scheduled': False, 'plan': str(RUN / 'run_plan.json'),
            'sha256': sha(RUN / 'run_plan.json'), 'only_parameter_change': {'alpha': [1., .5]}}))
    elif args.command == 'verify':
        print(json.dumps(verify()))
    else:
        assert args.gpu_authorized and args.allocation_note, 'Root allocation and explicit flag required; no auto launch'
        run(args.allocation_note)


if __name__ == '__main__':
    main()
