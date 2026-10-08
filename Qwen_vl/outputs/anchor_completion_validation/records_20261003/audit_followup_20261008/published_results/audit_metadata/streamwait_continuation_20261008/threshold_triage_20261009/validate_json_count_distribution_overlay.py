"""Pure CPU tests; frozen sources, original artifacts and queue state stay intact."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import json_count_distribution_overlay as overlay

OUT = Path(__file__).resolve().parent
DIAG = overlay.OUT / 'next_gap_diagnosis_20261008'
STABILITY = DIAG / 'stability'
CONTROLLER = DIAG / 'schedule_next_full_controls.py'
PLAN = STABILITY / 'next_text_full_streamwait.controller.plan.json'
ARMS = ('EADP_beta2', 'AZ_beta2')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text())


def rejects(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, RuntimeError):
        return True
    return False


def frozen_scores():
    plan = read(PLAN)
    with overlay.installed_json_count_overlay():
        controller = load(CONTROLLER, 'seed_cpu_frozen_controller')
        scores = {arm: controller.validate_arm(plan['jobs'][arm], plan) for arm in ARMS}
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-probe', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError('Use the actual unchanged Python 3.10 gate interpreter')
    if args.seed_probe:
        print(json.dumps(frozen_scores(), sort_keys=True, separators=(',', ':')))
        return

    plan = read(PLAN)
    protected = {str(overlay.FROZEN_WORKER), str(CONTROLLER), str(PLAN)}
    protected.update(str(path) for path in overlay.TARGET_SHA256)
    for arm in ARMS:
        protected.update(plan['jobs'][arm]['output_paths'].values())
    protected = sorted(path for path in protected if Path(path).is_file())
    before = {path: overlay.sha256(path) for path in protected}
    gates = {}
    worker = load(overlay.FROZEN_WORKER, 'unmodified_cpu_worker')
    controller = load(CONTROLLER, 'unmodified_cpu_controller')
    questions = [json.loads(line) for line in Path(plan['jobs']['EADP_beta2']['parameters']['question_file']).open()]
    exact = {}
    mismatch = {}
    for arm in ARMS:
        job = plan['jobs'][arm]
        paths = job['output_paths']
        fresh = worker.validate_and_score(paths['prediction'], paths['runtime'], questions, job['parameters'])
        stored = read(paths['score'])
        mismatch[arm] = [key for key, value in fresh.items() if stored.get(key) != value]
        gates[arm + '_only_raw_mismatch_is_count_key_type'] = mismatch[arm] == ['actual_count_distribution']
        gates[arm + '_original_gate_reproduces_refusal'] = rejects(controller.validate_arm, job, plan)
        normalized = overlay.normalize_count_keys(fresh)
        gates[arm + '_strict_exact_score_and_every_return_field_equal'] = all(stored.get(key) == value for key, value in normalized.items())
        gates[arm + '_no_other_return_field_changed'] = all(normalized[key] == value for key, value in fresh.items() if key != 'actual_count_distribution')
        exact[arm] = normalized

    original_spec = importlib.util.spec_from_file_location
    with overlay.installed_json_count_overlay():
        wrapped = load(overlay.FROZEN_WORKER, 'wrapped_cpu_worker')
        unrelated = load(overlay.ROOT / 'LLaVA/llava/eval/m4c_evaluator.py', 'unchanged_cpu_m4c')
        gates['unrelated_module_is_unwrapped'] = not hasattr(unrelated.TextVQAAccuracyEvaluator.eval_pred_list, '_json_count_distribution_only_overlay')
        for arm in ARMS:
            job = plan['jobs'][arm]
            gates[arm + '_real_controller_all_original_gates_pass'] = controller.validate_arm(job, plan)['accuracy_percent'] == exact[arm]['accuracy_percent']
        gates['short_denominator_refused_by_original_validator'] = rejects(wrapped.validate_and_score,
            plan['jobs']['EADP_beta2']['output_paths']['prediction'], plan['jobs']['EADP_beta2']['output_paths']['runtime'],
            questions[:-1], plan['jobs']['EADP_beta2']['parameters'])
        with tempfile.TemporaryDirectory(prefix='eadp_json_count_cpu_') as temporary:
            runtime = Path(plan['jobs']['EADP_beta2']['output_paths']['runtime'])
            lines = runtime.read_text().splitlines()
            altered = json.loads(lines[0]); altered['actual_visual_tokens_retained'] = 161
            lines[0] = json.dumps(altered)
            invalid = Path(temporary) / 'invalid_runtime.jsonl'
            invalid.write_text('\n'.join(lines) + '\n')
            gates['out_of_budget_count_refused_by_original_validator'] = rejects(wrapped.validate_and_score,
                plan['jobs']['EADP_beta2']['output_paths']['prediction'], invalid, questions,
                plan['jobs']['EADP_beta2']['parameters'])

        original_read = controller.read
        score_path = Path(plan['jobs']['EADP_beta2']['output_paths']['score'])
        original_stored = read(score_path)
        for label, mutation in (
            ('changed_distribution_refused', lambda value: value['actual_count_distribution'].__setitem__('158', value['actual_count_distribution']['158'] + 1)),
            ('changed_denominator_refused', lambda value: value.__setitem__('n', 4999)),
            ('tiny_changed_accuracy_refused_without_tolerance', lambda value: value.__setitem__('accuracy_percent', value['accuracy_percent'] + 1e-10)),
        ):
            changed = copy.deepcopy(original_stored); mutation(changed)
            def changed_read(path, changed=changed):
                return changed if Path(path) == score_path else original_read(path)
            controller.read = changed_read
            try:
                gates[label] = rejects(controller.validate_arm, plan['jobs']['EADP_beta2'], plan)
            finally:
                controller.read = original_read
    gates['import_hook_restored'] = importlib.util.spec_from_file_location is original_spec

    original_sha = overlay.FROZEN_WORKER_SHA256
    overlay.FROZEN_WORKER_SHA256 = '0' * 64
    try:
        gates['wrong_pinned_worker_SHA_refused'] = rejects(overlay.check_worker_identity)
    finally:
        overlay.FROZEN_WORKER_SHA256 = original_sha
    gates['count_key_collision_refused'] = rejects(overlay.normalize_count_keys, {'actual_count_distribution': {1: 2, '1': 3}})
    gates['unapproved_CLI_target_refused'] = rejects(overlay.checked_target, str(overlay.FROZEN_WORKER))
    gates['relative_CLI_target_refused'] = rejects(overlay.checked_target, 'publish_repair_results.py')
    for target in overlay.TARGET_SHA256:
        gates['allowlisted_target_SHA_' + target.name] = overlay.checked_target(str(target)) == target

    probes = {}
    for seed in ('0', '731'):
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--seed-probe'],
            env=dict(os.environ, PYTHONHASHSEED=seed), check=True, capture_output=True, text=True)
        probes[seed] = json.loads(result.stdout)
    gates['two_hashseeds_same_full_strict_CPU_scores'] = probes['0'] == probes['731']
    after = {path: overlay.sha256(path) for path in protected}
    gates['all_original_sources_and_artifacts_unchanged'] = before == after
    gates['no_production_model_or_torch_imported'] = not any(name == 'torch' or name == 'llava' or name.startswith('llava.') for name in sys.modules)
    report = dict(success=all(gates.values()), CPU_only=True, production_modified=False, GPU_started=False,
        python=sys.version, python_executable=sys.executable, overlay_sha256=overlay.sha256(overlay.__file__),
        frozen_worker_sha256=original_sha, raw_mismatching_keys=mismatch,
        gates=gates, gate_count=len(gates), exact_full_scores=exact, hashseed_probes=probes,
        protected_before_sha256=before, protected_after_sha256=after,
        unchanged_scope='Return-key representation only; all score values and frozen exact gates remain unchanged')
    if not report['success']:
        raise AssertionError(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        if args.output.exists():
            raise FileExistsError('Preserve previous CPU validation output')
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(dict(success=report['success'], gates=len(gates), exact_scores={arm:value['accuracy_percent'] for arm,value in exact.items()}), indent=2))


if __name__ == '__main__':
    main()
