#!/usr/bin/env python3
"""Validate source-derived formulas on existing CPU arrays; never load a model."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def main():
    import numpy as np
    import torch
    from formula_helpers import bind_source, importance, quantile_filter, traced_selection, source_parts
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    plan = json.loads((HERE / 'run_plan.json').read_text())
    helper, source, eq10, traced, proof = bind_source(torch, Path(plan['arch_path']))
    cache = np.load(HERE.parent / 'cpu_formula_real_panel/cross_modal_intermediates.npz', allow_pickle=False)
    prior = json.loads((HERE.parent / 'cpu_formula_real_panel/real_formula_comparison.json').read_text())
    tested, old_importance_matches, eq10_matches = 0, 0, 0
    for i in range(16):
        global_sim, dense = torch.from_numpy(cache['global_%02d' % i]), torch.from_numpy(cache['dense_%02d' % i])
        for recipe in plan['recipes']:
            a, b = recipe['alpha'], recipe['beta']
            baseline, _ = importance(torch, helper, source, eq10, global_sim, dense, a, b, 'source')
            identity, _ = importance(torch, helper, source, eq10, global_sim, dense, a, b, 'source_members_order_identity')
            quantile, detail = importance(torch, helper, source, eq10, global_sim, dense, a, b, 'eq6_quantile_membership_only')
            numerator, _ = importance(torch, helper, source, eq10, global_sim, dense, a, b, 'eq10_numerator_only')
            assert torch.equal(identity, baseline)
            assert detail['positions'][:15] == detail['source_positions'] and len(detail['positions']) == 16
            assert len(set(detail['positions'])) == len(detail['positions'])
            if a == 1.:
                assert torch.equal(quantile, baseline)
            key = f'alpha{a}_beta{b}'
            if key in prior['rows'][i]['recipes']:
                old = prior['rows'][i]['recipes'][key]['variants']['Eq10_remove_numerator_epsilon_only']
                assert hashlib.sha256(baseline.numpy().tobytes()).hexdigest() == old['source_importance_sha256']
                assert hashlib.sha256(numerator.numpy().tobytes()).hexdigest() == old['variant_importance_sha256']
                old_importance_matches += 1; eq10_matches += 1
            tested += 1
    # Exact entropy ties retain all77 positions, preserving the original top15 prefix.
    dense = torch.zeros(1, 576, 1, 77)
    _, _, info = quantile_filter(torch, dense, T=100., entropy_keep_ratio=.2)
    assert len(info['positions']) == 77 and info['ties_at_threshold'] == 77
    assert info['positions'][:15] == info['source_positions']
    # CPU FP32 synthetic fixture tests observer equivalence only, never real facility behavior.
    helper.visual_token_num = 3
    generator = torch.Generator().manual_seed(20261028)
    sim = torch.rand(1, 7, 7, generator=generator)
    values = torch.rand(1, 7, generator=generator)
    original_ids, original_quotas = helper.greed_select(values, sim, K=1, strategy='importance_based')
    observed_ids, observed_quotas, gains = traced_selection(torch, helper, traced, values, sim)
    assert torch.equal(original_ids, observed_ids) and torch.equal(original_quotas, observed_quotas)
    assert gains.shape == (observed_ids.shape[1], 1, 7)
    # Eq10 proof checks every denominator AST, rather than only the changed numerator.
    import ast
    denom = lambda module: [ast.dump(n.right, include_attributes=False) for n in ast.walk(module)
                            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)]
    assert denom(source) == denom(eq10) and proof['eq10_restoration_identical']
    result = subprocess.run([sys.executable, str(HERE / 'diagnose_facility.py')], text=True, capture_output=True, check=True)
    default = json.loads(result.stdout)
    assert default['GPU_started'] is False and default['torch_imported'] is False
    assert not torch.cuda.is_initialized()
    report = {'status': 'passed', 'existing_CPU_panel_formula_cases': tested,
              'source_importance_exact_matches_previous': old_importance_matches,
              'eq10_importance_exact_matches_previous': eq10_matches,
              'source_members_and_order_identity_controls': 48,
              'alpha1_quantile_exact_negative_controls': 16,
              'quantile_all77_ties_fixture_passed': True, 'quantile_original_prefix_order_preserved': True,
              'all_importance_denominator_ASTs_identical': True,
              'greedy_readonly_trace_CPU_synthetic_equivalence': True,
              'default_entry_CPU_only_no_tensor_runtime_import': default,
              'actual_real_facility_sets_measured': False, 'CUDA_initialized': False,
              'limitations': 'CPU fixtures validate code/proofs, not native GPU projected features or facility outcome.'}
    (HERE / 'cpu_acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
