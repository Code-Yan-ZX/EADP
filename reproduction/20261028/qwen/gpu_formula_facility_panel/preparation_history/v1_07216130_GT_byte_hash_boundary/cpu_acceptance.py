"""Synthetic arithmetic and CPU-default tests only; never a native feature claim."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch
    from formula_helpers import bind, parts, quantile_filter, RaggedQuantile
    from diagnose import sha, save, validate
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    plan = json.loads((HERE / 'run_plan.json').read_text())
    spec = importlib.util.spec_from_file_location('original_Qwen_pruner_CPU', plan['pruner_path'])
    native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
    compute, select, proof = bind(torch, native, plan['pruner_path'])
    generator = torch.Generator().manual_seed(20261028)
    features = torch.randn(1, 64, 32, generator=generator).to(torch.bfloat16)
    seq = torch.randn(1, 21, 32, generator=generator).to(torch.bfloat16); global_text = seq.mean(1)
    pruner = native.VisualTokenPruner(visual_token_num=16, alpha=.5, beta=2.)
    base = pruner.compute_importance(features, global_text, seq, 8, 8)
    sim = native._sim_visual_impl(features)
    source_ids = None; source_gains = None
    for variant in plan['variants']:
        value, details = compute(pruner, features, global_text, seq, 8, 8, variant)
        ids, quotas, gains = select(value, sim, 16)
        assert quotas.tolist() == [16] and torch.isfinite(value).all() and torch.isfinite(gains).all()
        assert value.dtype == sim.dtype == torch.float32
        if variant == 'source':
            assert torch.equal(value, base); source_ids = ids; source_gains = gains
        if variant == 'source_members_order_identity':
            assert torch.equal(value, base) and torch.equal(ids, source_ids) and torch.equal(gains, source_gains)
        if variant == 'eq6_quantile_membership_only':
            assert all(r['positions'][:len(r['source_positions'])] == r['source_positions'] for r in details['rows'])
    _, source, changed, _, prefix, _ = parts(plan['pruner_path'])
    dense = torch.zeros(1, 64, 1, 21)
    _, _, ties = quantile_filter(torch, prefix, dense)
    assert ties['lengths'] == [21] and ties['rows'][0]['ties_at_threshold'] == 21
    nonconstant = torch.linspace(-1, 1, 64).view(1, 64, 1, 1) * torch.linspace(.01, .8, 21).view(1, 1, 1, 21)
    mixed = torch.cat([dense, nonconstant], dim=0)
    try:
        quantile_filter(torch, prefix, mixed)
        raise AssertionError('Ragged threshold must be explicitly unmeasured')
    except RaggedQuantile as error:
        assert not error.details['all_rows_same_length'] and not error.details['padding_or_rowwise_reduction_used']
    denominators = lambda code: [ast.dump(n.right, include_attributes=False) for n in ast.walk(code)
                                 if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)]
    assert denominators(source) == denominators(changed)
    result = subprocess.run([sys.executable, '-B', str(HERE / 'run_panel.py'), 'check'], check=True, text=True, capture_output=True)
    default = json.loads(result.stdout)
    assert not default['GPU_started'] and not default['container_started'] and not default['torch_imported']
    assert not torch.cuda.is_initialized()
    report = dict(status='passed', synthetic_CPU_formula_cases=4, model_loads=0, CUDA_initialized=False,
        source_identity_and_gain_exact=True, source_BF16_visual_norm_path=True, dynamic_L21_not77=True,
        source_members_and_order_retained=True, all_ties_preserved=True, ragged_explicit_unmeasured=True,
        greedy_trace_matches_unmodified=True, Eq10_denominators_unchanged=True, default=default,
        limitations='Synthetic CPU tensors verify code contracts only; actual native16 anchors/features/results remain unmeasured.')
    save(HERE / 'cpu_acceptance.json', report)
    print(json.dumps(report))


if __name__ == '__main__': main()
