#!/usr/bin/env python3
"""Synthetic CPU arithmetic/guard acceptance only; no model or feature extraction."""
import ast
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def main():
    import torch
    from anyres_helpers import common, importance, quantile_filter, effective_mask, traced_select, RaggedQuantile, source_mask_ast
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    plan = json.loads((HERE / 'run_plan.json').read_text())
    helper, source, eq10, traced, proof = common.bind_source(torch, Path(plan['arch_path']))
    dense = (torch.linspace(-1, 1, 576).view(1, 576, 1, 1)
             * torch.linspace(.01, .8, 77).view(1, 1, 1, 77)).expand(5, -1, 2, -1).contiguous()
    global_sim = torch.linspace(-.9, .9, 5 * 576 * 2).view(5, 576, 2)
    cases = 0
    for recipe in plan['recipes']:
        base, _ = importance(torch, helper, source, eq10, global_sim, dense, recipe['beta'], 'source')
        identity, _ = importance(torch, helper, source, eq10, global_sim, dense, recipe['beta'], 'source_members_order_identity')
        assert torch.equal(base, identity)
        quantile, details = importance(torch, helper, source, eq10, global_sim, dense, recipe['beta'], 'eq6_quantile_membership_only')
        numerator, _ = importance(torch, helper, source, eq10, global_sim, dense, recipe['beta'], 'eq10_numerator_only')
        assert all(row['positions'][:15] == row['source_positions'] for row in details['rows'])
        assert set(details['lengths']) == {16} and not details['padding_or_rowwise_aggregation_used']
        local, entropy = helper.entrpy_filter(dense, T=100., entropy_keep_ratio=.2)
        local = helper.local_aggregation(local, entropy, M_temp=.01, strategy='negative_entropy')
        raw = (.5 * global_sim + .5 * local).mean(-1).view(1, 2880)
        normalized = (raw - raw.min(-1, keepdim=True).values + 1e-6) / (raw.max(-1, keepdim=True).values - raw.min(-1, keepdim=True).values + 1e-6)
        expected = helper.spatial_smoothing(normalized.view(5, 576), kernel_size=3, sigma=1.) ** recipe['beta']
        assert torch.equal(base, expected)
        # Wrong cropwise min/max must fail this fixture; no independent-crop normalization is allowed.
        crop = raw.view(5, 576)
        wrong = (crop - crop.min(-1, keepdim=True).values + 1e-6) / (crop.max(-1, keepdim=True).values - crop.min(-1, keepdim=True).values + 1e-6)
        assert not torch.equal(normalized.view(5, 576), wrong)
        assert torch.isfinite(quantile).all() and torch.isfinite(numerator).all()
        cases += 4
    _, _, ties = quantile_filter(torch, torch.zeros_like(dense), 100., .2)
    assert set(ties['lengths']) == {77}
    ragged = dense.clone(); ragged[0, :, 0, :] = 0
    try:
        quantile_filter(torch, ragged, 100., .2)
        raise AssertionError('Ragged fixture must never be silently padded')
    except RaggedQuantile as error:
        assert not error.details['all_rows_same_length'] and set(error.details['lengths']) == {16, 77}
        assert not error.details['padding_or_rowwise_aggregation_used']
    # Verify original source greedy/trace/create-mask on a small CPU-only fixture.
    helper.visual_token_num = 3
    weights = torch.arange(1, 36, dtype=torch.float32).view(5, 7)
    generator = torch.Generator().manual_seed(20261028)
    sim = torch.rand(5, 7, 7, generator=generator)
    ids, quotas = helper.greed_select(weights, sim, K=5, strategy='importance_based')
    observed, observed_quota, gains = traced_select(torch, helper, traced, weights, sim)
    assert torch.equal(ids, observed) and torch.equal(quotas, observed_quota)
    mask = effective_mask(torch, ids, quotas)
    for crop in range(5):
        actual = mask[crop].nonzero(as_tuple=False).flatten().tolist()
        assert actual == sorted(ids[crop, :int(quotas[crop])].tolist())
    # Changes confined to discarded max_T suffixes cannot be called input changes.
    modified = ids.clone()
    for crop in range(5):
        for step in range(int(quotas[crop]), ids.shape[1]):
            modified[crop, step] = 100 + step  # outside valid prefix, unique unused locations
    assert not torch.equal(modified, ids) and torch.equal(effective_mask(torch, modified, quotas), mask)
    denominators = lambda code: [ast.dump(n.right, include_attributes=False) for n in ast.walk(code)
                                 if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)]
    assert denominators(source) == denominators(eq10)
    assert common.digest(source_mask_ast()) == plan['create_mask_ast_sha256']
    result = subprocess.run([sys.executable, str(HERE / 'diagnose_anyres.py')], check=True, text=True, capture_output=True)
    default = json.loads(result.stdout)
    assert not default['GPU_started'] and not default['torch_imported']
    assert not torch.cuda.is_initialized()
    report = {'status': 'passed', 'CPU_only_synthetic_formula_cases': cases,
              'model_loads': 0, 'CPU_or_GPU_real_feature_extractions': 0, 'CUDA_initialized': False,
              'source_identity_exact': True, 'cross_crop_global2880_normalization_verified': True,
              'incorrect_per_crop_normalization_rejected': True, 'uniform_quantile_prefix_order': True,
              'all77ties_preserved': True, 'ragged_explicit_unmeasured_exception': True,
              'no_padding_or_rowwise_reduce_substitution': True, 'source_greedy_trace_exact': True,
              'source_mask_AST_exact': True, 'discarded_suffix_only_changes_leave_mask_unchanged': True,
              'Eq10_all_denominators_unchanged': True, 'default_CPU_guard': default,
              'limitations': 'Code acceptance uses synthetic FP32 CPU tensors, not native NeXT features or facility outcomes.'}
    (HERE / 'cpu_acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
