#!/usr/bin/env python3
"""Validate the control registration and reject incomplete/corrupted evidence."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from alpha_control import RUN, validate_frozen
from verify_complete_predictions import verify


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    plan = json.loads((RUN / 'run_plan.json').read_text())
    base = json.loads((ROOT / 'run02_source_wait_only_k32/run_plan.json').read_text())
    outcomes = []
    validate_frozen(plan)
    outcomes.append({'check': 'frozen_source_inputs_500images_model17files_implementation', 'passed': True})
    delta = {key: [base['parameters'].get(key), value] for key, value in plan['parameters'].items() if value != base['parameters'].get(key)}
    assert delta == {'alpha': [1., .5]}
    outcomes.append({'check': 'exact_single_parameter_delta', 'passed': True, 'delta': delta})
    altered = copy.deepcopy(plan)
    altered['source_files']['LLaVA/llava/model/llava_arch.py'] = '0' * 64
    try:
        validate_frozen(altered)
        raise RuntimeError('mutated source identity accepted')
    except AssertionError:
        outcomes.append({'check': 'mutated_source_hash_rejected', 'passed': True})
    rows = [json.loads(x) for x in (ROOT / 'run02_source_wait_only_k32/predictions.jsonl').read_text().splitlines()]
    with tempfile.TemporaryDirectory(prefix='cpu_acceptance_', dir=RUN) as tmp:
        temporary = Path(tmp)
        for name in ['partial', 'duplicate', 'reordered', 'prompt_changed']:
            fixture = copy.deepcopy(rows)
            if name == 'partial':
                fixture.pop()
            elif name == 'duplicate':
                fixture[-1] = fixture[0]
            elif name == 'reordered':
                fixture[0], fixture[1] = fixture[1], fixture[0]
            else:
                fixture[0]['prompt'] += ' altered'
            path = temporary / (name + '.jsonl')
            path.write_text(''.join(json.dumps(x) + '\n' for x in fixture))
            try:
                verify(path, temporary / name)
                raise RuntimeError(name + ' unexpectedly accepted')
            except AssertionError:
                outcomes.append({'check': name + '_evidence_rejected', 'passed': True})
    reference = verify(ROOT / 'run02_source_wait_only_k32/predictions.jsonl', RUN / 'cpu_reference_verification')
    assert reference['n_predictions'] == 8910 and abs(reference['official_macro_f1_percent'] - 82.22023699908897) < 1e-12
    outcomes.append({'check': 'real_full8910_reference_passes_official_and_independent', 'passed': True,
        'F1': reference['official_macro_f1_percent']})
    sys.path.insert(0, plan['generation_cwd'])
    import torch
    from wait_only_patch import install
    assert not torch.cuda.is_initialized()
    proof = install()
    expected = plan['wait_patch_prepared_proof']
    assert proof['patched_forward_ast_sha256'] == expected['patched_forward_ast_sha256']
    assert proof['restoration_ast_identical'] and proof['runtime_patch_installed']
    assert not torch.cuda.is_initialized()
    outcomes.append({'check': 'actual_fresh_import_three_wait_AST_identity_CPU_only', 'passed': True})
    for name in ['alpha_control.py', 'worker_alpha_control.py', 'verify_complete_predictions.py']:
        compile((ROOT / name).read_text(), str(ROOT / name), 'exec')
    outcomes.append({'check': 'worker_controller_verifier_compile_without_execution', 'passed': True})
    report = {'checks': outcomes, 'all_passed': all(x['passed'] for x in outcomes),
        'cuda_initialized': torch.cuda.is_initialized(), 'gpu_model_loaded': False,
        'state': json.loads((RUN / 'state.json').read_text())['status'],
        'scope': 'CPU registration/integrity acceptance only. Negative fixtures are temporary and never benchmark outputs.'}
    (RUN / 'cpu_acceptance_checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'checks_passed': len(outcomes), 'all_passed': report['all_passed'], 'cuda_initialized': False,
        'state': report['state']}))


if __name__ == '__main__':
    main()
