#!/usr/bin/env python3
"""CPU-only terminal audit of frozen native GPU diagnostic artifacts."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OUT = HERE / 'execution'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stat_guard(path):
    p = Path(path); s = p.stat()
    return dict(resolved=str(p.resolve()), bytes=s.st_size, device=s.st_dev, inode=s.st_ino,
                mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def main():
    import numpy as np
    assert not (OUT / 'failure.json').exists()
    plan = json.loads((HERE / 'run_plan.json').read_text())
    summary = json.loads((OUT / 'summary.json').read_text())
    identity = json.loads((OUT / 'process_identity.json').read_text())
    runtime = json.loads((OUT / 'loaded_runtime.json').read_text())
    allocation = json.loads((HERE / 'root_allocation.json').read_text())
    assert summary['status'] == 'complete' and summary['n_images'] == 16 and summary['cases'] == 192
    assert summary['plan_sha256'] == identity['plan_sha256'] == allocation['run_plan_sha256'] == sha(HERE / 'run_plan.json')
    assert summary['per_image_sha256'] == sha(OUT / 'per_image.jsonl')
    assert summary['loaded_runtime_sha256'] == sha(OUT / 'loaded_runtime.json')
    assert identity == runtime['identity']
    assert identity['allocation_note_sha256'] == sha(HERE / 'root_allocation.json')
    assert sha(OUT / 'allocation_note.json') == sha(HERE / 'root_allocation.json')
    proc = Path('/proc') / str(identity['pid'])
    same_worker = proc.exists() and int((proc / 'stat').read_text().split()[21]) == identity['start_ticks']
    assert not same_worker, 'Wait until the actual original worker exits'
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                         check=True, text=True, capture_output=True)
    gpu_pids = {int(x.split(',')[0]) for x in gpu.stdout.splitlines() if x.strip()}
    assert identity['pid'] not in gpu_pids
    source_guards = {}
    for group in ['source_sha256', 'implementation_sha256', 'reference_sha256']:
        source_guards[group] = len(plan[group])
        assert all(sha(path) == value for path, value in plan[group].items())
    for path, expected in plan['model_stat_guards'].items():
        assert stat_guard(path) == {k: v for k, v in expected.items() if k != 'prior_verified_sha256'}
    assert all(sha(row['image_path']) == row['image_sha256'] for row in plan['panel'])
    assert runtime['dtype'] == {'vision': ['torch.float32'], 'text': ['torch.float32'],
                                'projection': ['torch.float32'], 'mm_projector': ['torch.float16']}
    assert runtime['attention_backend'] == 'sdpa' and not runtime['autocast_enabled']
    assert not runtime['TF32_matmul'] and runtime['default_dtype'] == 'torch.float32'
    assert runtime['language_model_generation_calls'] == 0
    rows = [json.loads(line) for line in (OUT / 'per_image.jsonl').read_text().splitlines()]
    assert len(rows) == 16 and [r['index'] for r in rows] == list(range(16))
    aggregates = {}; artifacts = []; n_cases = 0; finite_arrays = 0; max_objective_delta = 0.
    first_divergences = []
    for index, row in enumerate(rows):
        assert row['panel'] == plan['panel'][index] and row['native_feature_extractions'] == 1
        assert row['native_source_anchor_matches']
        path = Path(row['cache_path'])
        assert path == OUT / ('image_%02d.npz' % index) and path.stat().st_size == row['cache_bytes']
        assert sha(path) == row['cache_sha256']
        artifacts.append({'path': str(path), 'sha256': row['cache_sha256'], 'bytes': row['cache_bytes']})
        with np.load(path, allow_pickle=False) as cache:
            assert len(cache.files) == 56
            for key in cache.files:
                assert np.isfinite(cache[key]).all(), (index, key)
                finite_arrays += 1
            for key, shape, dtype in [
                    ('projected_fp16', (1, 576, 4096), np.float16),
                    ('visual_sim_fp32', (1, 576, 576), np.float32),
                    ('global_fp32', (1, 576, 1), np.float32),
                    ('dense_fp32', (1, 576, 1, 77), np.float32),
                    ('input_fp16', (1, 3, 336, 336), np.float16)]:
                assert cache[key].shape == shape and cache[key].dtype == dtype, (key, cache[key].shape, cache[key].dtype)
            assert np.array_equal(cache['native_importance'], cache['alpha1_beta1__source__importance'])
            assert np.array_equal(cache['native_selected_ids'], cache['alpha1_beta1__source__ids'])
            assert np.array_equal(cache['native_final_mask'], cache['alpha1_beta1__source__final_mask'])
            assert int(cache['native_final_mask'].sum()) == 32
            expected_cases = {(recipe['name'], variant) for recipe in plan['recipes'] for variant in plan['variants']}
            assert len(row['cases']) == 12
            assert {(r['recipe']['name'], r['variant']) for r in row['cases']} == expected_cases
            for case in row['cases']:
                recipe, variant = case['recipe']['name'], case['variant']
                assert case['recipe'] == next(r for r in plan['recipes'] if r['name'] == recipe)
                key = recipe + '__' + variant
                imp, ids, gains, mask = (cache[key + '__' + name] for name in ['importance', 'ids', 'gain_trace', 'final_mask'])
                base = recipe + '__source'
                base_imp, base_ids, base_gains = (cache[base + '__' + name] for name in ['importance', 'ids', 'gain_trace'])
                assert imp.shape == (1, 576) and ids.shape == (1, 32) and gains.shape == (32, 1, 576)
                assert imp.dtype == np.float32 and gains.dtype == np.float32 and ids.dtype == np.int64
                assert mask.dtype == np.bool_ and mask.shape == (1, 576)
                selected, original = ids[0].tolist(), base_ids[0].tolist()
                assert len(set(selected)) == 32 and all(0 <= x < 576 for x in selected)
                assert selected == case['selected_ids_ordered'] and original == case['baseline_ids_ordered']
                assert case['final_retained_ids_spatial_order'] == sorted(selected)
                expected_mask = np.zeros((1, 576), dtype=np.bool_); expected_mask[0, selected] = True
                assert np.array_equal(expected_mask, mask) and int(mask.sum()) == 32 and case['quota'] == [32]
                assert np.argmax(gains[:, 0, :], axis=1).tolist() == selected
                for step in range(32):
                    assert np.all(gains[step, 0, selected[:step]] == np.float32(-1e9))
                top2 = np.sort(gains[:, 0, :], axis=-1)[:, -2:]
                margins = top2[:, 1] - top2[:, 0]
                assert margins.tolist() == case['greedy_step_top2_gain_margins']
                ties = (gains == gains.max(axis=-1, keepdims=True)).sum(axis=-1).ravel()
                assert ties.tolist() == case['exact_gain_ties_per_step']
                unchanged = np.array_equal(imp, base_imp)
                delta = float(np.abs(imp - base_imp).max())
                overlap = len(set(selected) & set(original))
                first = next((i for i, (x, y) in enumerate(zip(selected, original)) if x != y), None)
                assert unchanged == case['importance_exact_equal'] and delta == case['importance_max_abs_delta']
                assert overlap == case['selected_set_overlap'] and (first is None) == case['ordered_sequence_equal']
                assert first == case['first_divergence_step_zero_based']
                if first is None:
                    assert case['first_divergence_comparison'] is None
                else:
                    first_divergences.append({'image_index': index, 'recipe': recipe, 'variant': variant,
                                              'first_step': first, 'gains': case['first_divergence_comparison']})
                if variant == 'source_members_order_identity' or (variant == 'eq6_quantile_membership_only' and recipe == 'alpha1_beta1'):
                    assert unchanged and np.array_equal(ids, base_ids) and np.array_equal(gains, base_gains)
                if variant == 'eq6_quantile_membership_only':
                    detail = case['text_filter']
                    assert detail['positions'][:15] == detail['source_positions']
                    assert len(detail['positions']) == 16 and detail['source_prefix_order_preserved']
                # Independent weighted-coverage check uses the saved real GPU matrix.
                # NumPy FP64 summation is a verification tolerance, never replacement selection arithmetic.
                for selected_set, weights, name in [
                        (original, base_imp, 'baseline_objective_baseline_selection'),
                        (selected, base_imp, 'baseline_objective_variant_selection'),
                        (original, imp, 'variant_objective_baseline_selection'),
                        (selected, imp, 'variant_objective_variant_selection')]:
                    coverage = np.maximum(cache['visual_sim_fp32'][:, selected_set, :].max(axis=1), np.float32(0.))
                    products = coverage * weights
                    independent = float(products.sum(dtype=np.float64))
                    error = abs(independent - case[name]); max_objective_delta = max(max_objective_delta, error)
                    assert error <= max(1e-5, 4 * np.finfo(np.float32).eps * 576 * float(np.abs(products).sum(dtype=np.float64)))
                key2 = (recipe, variant)
                agg = aggregates.setdefault(key2, {'recipe': recipe, 'variant': variant, 'n_images': 0,
                    'importance_changed_images': 0, 'ordered_sequence_changed_images': 0, 'retained_set_changed_images': 0,
                    'minimum_retained_set_overlap': 32, 'max_importance_abs_delta': 0.,
                    'max_abs_greedy_gain_perturbation': 0., 'minimum_source_top2_gain_margin': float('inf')})
                agg['n_images'] += 1; agg['importance_changed_images'] += not unchanged
                agg['ordered_sequence_changed_images'] += first is not None; agg['retained_set_changed_images'] += overlap != 32
                agg['minimum_retained_set_overlap'] = min(agg['minimum_retained_set_overlap'], overlap)
                agg['max_importance_abs_delta'] = max(agg['max_importance_abs_delta'], delta)
                agg['max_abs_greedy_gain_perturbation'] = max(agg['max_abs_greedy_gain_perturbation'], float(np.abs(gains - base_gains).max()))
                base_top2 = np.sort(base_gains[:, 0, :], axis=-1)[:, -2:]
                agg['minimum_source_top2_gain_margin'] = min(agg['minimum_source_top2_gain_margin'], float((base_top2[:, 1] - base_top2[:, 0]).min()))
                n_cases += 1
    assert n_cases == 192 and finite_arrays == 896
    for expected in summary['aggregate']:
        actual = aggregates[(expected['recipe'], expected['variant'])]
        assert all(actual[k] == v for k, v in expected.items())
    assert 'torch' not in sys.modules, 'The independent audit must remain CPU-only without torch'
    report = {'success': True, 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'n_images': 16, 'n_cases': n_cases, 'npz_artifacts': artifacts, 'finite_arrays_checked': finite_arrays,
              'source_sha_guard_counts': source_guards, 'model_stat_guards_checked': len(plan['model_stat_guards']),
              'image_sha_checked': 16, 'source_anchor_images_exact': 16,
              'case_ID_mask_trace_argmax_consistency': 192, 'identity_case_exact_controls': 48,
              'alpha1_eq6_exact_controls': 16, 'aggregate': list(aggregates.values()),
              'first_selection_divergences': first_divergences,
              'max_independent_weighted_coverage_sum_delta_from_GPU': max_objective_delta,
              'worker_exit': {'identity': identity, 'same_worker_still_alive': False, 'proc_path_exists': proc.exists(),
                              'gpu_pid_absent': True, 'nvidia_smi_stdout': gpu.stdout,
                              'PTY_exit_code': 'Owned and recorded by root exec49578; not inferred by this verifier'},
              'artifacts_sha256': {str(p): sha(p) for p in [HERE / 'run_plan.json', HERE / 'root_allocation.json',
                                    OUT / 'process_identity.json', OUT / 'loaded_runtime.json', OUT / 'per_image.jsonl',
                                    OUT / 'summary.json', HERE / 'diagnostic.log', Path(__file__)]},
              'benchmark_answers_read': False, 'benchmark_scores_computed': False, 'GPU_started': False,
              'limitations': 'Real native GPU saved tensors and traces verified. No CPU projector or GPU re-execution. Fixed16 only; no claim of benchmark accuracy, decoder schedule replay or paper recipe identity.'}
    (HERE / 'independent_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in {'npz_artifacts', 'artifacts_sha256'}}))


if __name__ == '__main__':
    main()
