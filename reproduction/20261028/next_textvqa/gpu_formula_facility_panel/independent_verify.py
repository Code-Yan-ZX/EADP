#!/usr/bin/env python3
"""CPU-only terminal validation of NeXT real AnyRes diagnostic artifacts."""
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
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def stat_guard(path):
    path = Path(path); s = path.stat()
    return dict(resolved=str(path.resolve()), bytes=s.st_size, device=s.st_dev,
                inode=s.st_ino, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def main():
    import numpy as np
    assert not (OUT / 'failure.json').exists(), 'Preserve original failure, do not retry generation'
    plan = json.loads((HERE / 'run_plan.json').read_text())
    summary = json.loads((OUT / 'summary.json').read_text())
    identity = json.loads((OUT / 'process_identity.json').read_text())
    allocation = json.loads((OUT / 'allocation_note.json').read_text())
    runtime = json.loads((OUT / 'loaded_runtime.json').read_text())
    assert summary['status'] in {'complete', 'complete_with_unmeasured_ragged_eq6'}
    assert summary['plan_sha256'] == identity['plan_sha256'] == allocation['run_plan_sha256'] == sha(HERE / 'run_plan.json')
    assert identity['allocation_note_sha256'] == sha(OUT / 'allocation_note.json')
    assert identity == runtime['identity'] and summary['n_images'] == summary['source_anchor_images'] == 16
    assert summary['planned_case_slots'] == 320 and summary['identity_controls'] == 80
    assert summary['per_image_sha256'] == sha(OUT / 'per_image.jsonl')
    assert summary['loaded_runtime_sha256'] == sha(OUT / 'loaded_runtime.json')
    proc = Path('/proc') / str(identity['pid'])
    same_worker = proc.exists() and int((proc / 'stat').read_text().split()[21]) == identity['start_ticks']
    assert not same_worker, 'Original actual generation worker must exit before terminal validation'
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                         text=True, capture_output=True, check=True)
    assert identity['pid'] not in {int(line.split(',')[0]) for line in gpu.stdout.splitlines() if line.strip()}
    for group in ['source_sha256', 'implementation_sha256', 'reference_sha256']:
        assert all(sha(path) == expected for path, expected in plan[group].items()), group
    extra = allocation.get('additional_source_sha256', {})
    assert extra, 'Root allocation additionally freezes transitive native source guards'
    assert all(sha(path) == expected for path, expected in extra.items())
    for path, expected in plan['model_stat_guards'].items():
        assert stat_guard(path) == {k: v for k, v in expected.items() if k != 'prior_verified_sha256'}
    assert all(sha(row['image_path']) == row['image_sha256'] for row in plan['panel'])
    assert runtime['dtype'] == plan['expected_dtype'] and runtime['backend'] == 'sdpa'
    assert runtime['default_dtype'] == 'torch.float32' and not runtime['autocast_enabled'] and not runtime['matmul_TF32']
    assert (runtime['native_alpha'], runtime['native_beta'], runtime['native_K']) == (.5, 1., 128)
    assert runtime['language_model_generation_calls'] == 0 and runtime['image_aspect_ratio'] == 'anyres'
    rows = [json.loads(line) for line in (OUT / 'per_image.jsonl').read_text().splitlines()]
    assert len(rows) == 16 and [row['index'] for row in rows] == list(range(16))
    aggregates = {}; artifacts = []; measured = skipped = arrays_finite = identities = 0
    divergences = []; quota_floor_differences_from_FP64_reference = []
    for index, row in enumerate(rows):
        assert row['panel'] == plan['panel'][index] and row['native_feature_extractions'] == 1
        assert row['split_sizes'] == [5] and row['cross_crop_normalization_shape'] == [1, 2880]
        assert row['native_source_anchor_matches']
        path = Path(row['cache_path'])
        assert path == OUT / ('image_%02d.npz' % index) and sha(path) == row['cache_sha256']
        assert path.stat().st_size == row['cache_bytes']
        artifacts.append({'path': str(path), 'sha256': row['cache_sha256'], 'bytes': row['cache_bytes']})
        assert len(row['cases']) == 20
        assert {(c['recipe']['name'], c['variant']) for c in row['cases']} == {
            (recipe['name'], variant) for recipe in plan['recipes'] for variant in plan['variants']}
        with np.load(path, allow_pickle=False) as cache:
            n_valid = sum(c['status'] == 'measured' for c in row['cases'])
            assert len(cache.files) == 9 + 5 * n_valid
            for name in cache.files:
                assert np.isfinite(cache[name]).all(), (index, name)
                arrays_finite += 1
            for name, shape, dtype in [
                ('projected_fp16', (5, 576, 4096), np.float16), ('visual_sim_fp32', (5, 576, 576), np.float32),
                ('native_importance', (5, 576), np.float32), ('native_quotas', (5,), np.int64),
                ('native_final_mask', (5, 576), np.bool_), ('actual_input_fp16', (1, 5, 3, 336, 336), np.float16)]:
                assert cache[name].shape == shape and cache[name].dtype == dtype, (name, cache[name].shape, cache[name].dtype)
            dense_shape = tuple(row['actual_dense_shape'])
            assert dense_shape[:2] == (5, 576) and dense_shape[-1] == 77
            assert cache['dense_fp32'].shape == dense_shape and cache['dense_fp32'].dtype == np.float32
            assert cache['global_fp32'].shape == dense_shape[:-1] and cache['global_fp32'].dtype == np.float32
            anchor = 'K128_alpha05_beta1__source'
            for native, observed in [('native_importance', 'importance'), ('native_internal_ids', 'internal_ids'),
                                     ('native_quotas', 'quotas'), ('native_final_mask', 'final_mask')]:
                assert np.array_equal(cache[native], cache[anchor + '__' + observed])
            for case in row['cases']:
                recipe = case['recipe']; variant = case['variant']; key = recipe['name'] + '__' + variant
                assert recipe == next(r for r in plan['recipes'] if r['name'] == recipe['name'])
                aggregate = aggregates.setdefault((recipe['name'], variant), {'recipe': recipe['name'], 'variant': variant,
                    'case_slots': 0, 'measured': 0, 'unmeasured_ragged': 0, 'importance_changed_images': 0,
                    'quota_changed_images': 0, 'effective_prefix_changed_images': 0,
                    'actual_final_mask_changed_images': 0, 'internal_sequence_changed_images': 0,
                    'maximum_importance_abs_delta': 0.})
                aggregate['case_slots'] += 1
                if case['status'] == 'unmeasured_ragged_quantile':
                    assert variant == 'eq6_quantile_membership_only'
                    info = case['text_filter']
                    assert not info['all_rows_same_length'] and len(set(info['lengths'])) > 1
                    assert not info['padding_or_rowwise_aggregation_used'] and case['no_implicit_padding_or_reduction_change']
                    assert not any(name.startswith(key + '__') for name in cache.files)
                    aggregate['unmeasured_ragged'] += 1; skipped += 1
                    continue
                assert case['status'] == 'measured'
                weights, ids, quotas, mask, gains = (cache[key + '__' + n] for n in ['importance', 'internal_ids', 'quotas', 'final_mask', 'gain_trace'])
                base = recipe['name'] + '__source'
                bweights, bids, bquotas, bmask, bgains = (cache[base + '__' + n] for n in ['importance', 'internal_ids', 'quotas', 'final_mask', 'gain_trace'])
                max_T = int(quotas.max())
                assert weights.shape == (5, 576) and weights.dtype == np.float32
                assert quotas.shape == (5,) and quotas.dtype == np.int64 and np.all(quotas >= 1)
                assert ids.shape == (5, max_T) and ids.dtype == np.int64 and 0 < max_T <= 576
                assert gains.shape == (max_T, 5, 576) and gains.dtype == np.float32
                assert mask.shape == (5, 576) and mask.dtype == np.bool_
                assert case['quota'] == quotas.tolist() and case['baseline_quota'] == bquotas.tolist()
                assert int(quotas.sum()) <= recipe['K'] * 5
                assert np.argmax(gains, axis=2).T.tolist() == ids.tolist()
                expected_mask = np.zeros((5, 576), dtype=np.bool_)
                prefixes, spatial, bprefixes, bspatial = [], [], [], []
                for crop in range(5):
                    assert len(set(ids[crop].tolist())) == max_T and np.all((ids[crop] >= 0) & (ids[crop] < 576))
                    prefix = ids[crop, :int(quotas[crop])].tolist(); prefixes.append(prefix)
                    expected_mask[crop, prefix] = True; spatial.append(sorted(prefix))
                    bpref = bids[crop, :int(bquotas[crop])].tolist(); bprefixes.append(bpref); bspatial.append(sorted(bpref))
                    for t in range(max_T):
                        assert np.all(gains[t, crop, ids[crop, :t]] == np.float32(-1e9))
                assert np.array_equal(mask, expected_mask)
                assert np.array_equal(mask.sum(axis=1), quotas)
                assert case['actual_retained_count'] == int(mask.sum()) and case['baseline_retained_count'] == int(bmask.sum())
                assert case['effective_prefix_ids'] == prefixes and case['baseline_effective_prefix_ids'] == bprefixes
                assert case['retained_ids_spatial_order_by_crop'] == spatial and case['baseline_retained_ids_spatial_order_by_crop'] == bspatial
                assert case['retained_overlap_by_crop'] == [len(set(spatial[c]) & set(bspatial[c])) for c in range(5)]
                assert case['retained_added_by_crop'] == [sorted(set(spatial[c]) - set(bspatial[c])) for c in range(5)]
                assert case['retained_removed_by_crop'] == [sorted(set(bspatial[c]) - set(spatial[c])) for c in range(5)]
                assert case['effective_length_changes_by_crop'] == [
                    {'crop': c, 'baseline_quota': len(bprefixes[c]), 'variant_quota': len(prefixes[c])}
                    for c in range(5) if len(bprefixes[c]) != len(prefixes[c])]
                unchanged = np.array_equal(weights, bweights); delta = float(np.abs(weights - bweights).max())
                quota_equal = np.array_equal(quotas, bquotas); mask_equal = np.array_equal(mask, bmask)
                internal_equal = np.array_equal(ids, bids); prefix_equal = prefixes == bprefixes
                assert case['importance_exact_equal'] == unchanged and case['importance_max_abs_delta'] == delta
                assert case['quota_exact_equal'] == quota_equal and case['final_mask_exact_equal'] == mask_equal
                assert case['internal_maxT_sequence_equal'] == internal_equal and case['effective_prefix_order_equal'] == prefix_equal
                assert case['internal_discarded_changes_are_not_model_input_changes'] and case['traced_greedy_matches_unmodified']
                expected_divergences = []
                for crop in range(5):
                    t = next((t for t, (x, y) in enumerate(zip(prefixes[crop], bprefixes[crop])) if x != y), None)
                    if t is None: continue
                    picks = [bprefixes[crop][t], prefixes[crop][t]]
                    expected_divergences.append({'crop': crop, 'first_effective_step_zero_based': t,
                        'same_prefix_before_step': prefixes[crop][:t] == bprefixes[crop][:t],
                        'picks_baseline_variant': picks,
                        'baseline_gains_for_picks': [float(bgains[t, crop, k]) for k in picks],
                        'variant_gains_for_picks': [float(gains[t, crop, k]) for k in picks]})
                assert expected_divergences == case['first_effective_divergences']
                if expected_divergences or not quota_equal or not mask_equal:
                    divergences.append({'image_index': index, 'recipe': recipe['name'], 'variant': variant,
                        'quota_equal': quota_equal, 'final_mask_equal': mask_equal,
                        'effective_divergences': expected_divergences,
                        'added_by_crop': case['retained_added_by_crop'], 'removed_by_crop': case['retained_removed_by_crop']})
                if variant == 'source_members_order_identity':
                    assert unchanged and quota_equal and mask_equal and internal_equal and np.array_equal(gains, bgains)
                    identities += 1
                if variant == 'eq6_quantile_membership_only':
                    info = case['text_filter']
                    assert info['all_rows_same_length'] and len(set(info['lengths'])) == 1
                    assert not info['padding_or_rowwise_aggregation_used']
                    assert len(info['rows']) == 5 * dense_shape[2]
                    for textrow in info['rows']:
                        assert textrow['positions'][:15] == textrow['source_positions']
                        assert len(textrow['positions']) >= 16 and len(set(textrow['positions'])) == len(textrow['positions'])
                # Independent quota reference, not substituted source arithmetic.
                sums = weights.sum(axis=1, dtype=np.float64)
                quota64 = np.maximum(1, np.floor(sums / (sums.sum() + 1e-6) * recipe['K'] * 5)).astype(np.int64)
                if not np.array_equal(quota64, quotas):
                    # Saved source uses FP32 reduction/division; accept only the standard
                    # worst-case FP32 summation interval at integer rounding boundaries.
                    u = np.finfo(np.float32).eps / 2; gamma = (576 * u) / (1 - 576 * u)
                    low = np.maximum(0., sums - gamma * np.abs(weights).sum(axis=1, dtype=np.float64))
                    high = sums + gamma * np.abs(weights).sum(axis=1, dtype=np.float64)
                    lo = np.maximum(1, np.floor(low / (high.sum() + 1e-6) * recipe['K'] * 5 * (1 - 8 * u))).astype(np.int64)
                    hi = np.maximum(1, np.floor(high / (low.sum() + 1e-6) * recipe['K'] * 5 * (1 + 8 * u))).astype(np.int64)
                    assert np.all((quotas >= lo) & (quotas <= hi))
                    quota_floor_differences_from_FP64_reference.append({'image': index, 'recipe': recipe['name'],
                        'variant': variant, 'native_FP32': quotas.tolist(), 'FP64_reference': quota64.tolist(),
                        'FP32_interval_low': lo.tolist(), 'FP32_interval_high': hi.tolist()})
                aggregate['measured'] += 1; aggregate['importance_changed_images'] += not unchanged
                aggregate['quota_changed_images'] += not quota_equal; aggregate['effective_prefix_changed_images'] += not prefix_equal
                aggregate['actual_final_mask_changed_images'] += not mask_equal; aggregate['internal_sequence_changed_images'] += not internal_equal
                aggregate['maximum_importance_abs_delta'] = max(aggregate['maximum_importance_abs_delta'], delta)
                measured += 1
    assert measured + skipped == 320 and identities == 80
    assert summary['measured_cases'] == measured and summary['unmeasured_ragged_cases'] == skipped
    assert (summary['status'] == 'complete') == (skipped == 0), 'Ragged must never count as fully measured'
    for expected in summary['aggregate']:
        actual = aggregates[(expected['recipe'], expected['variant'])]
        assert all(actual[k] == v for k, v in expected.items())
    assert 'torch' not in sys.modules
    report = {'success': True, 'checked_utc': datetime.now(timezone.utc).isoformat(), 'n_images': 16,
        'planned_case_slots': 320, 'measured_cases': measured, 'unmeasured_ragged_cases': skipped,
        'all_planned_cases_measured': skipped == 0, 'finite_arrays_checked': arrays_finite,
        'source_anchor_images_exact': 16, 'identity_controls_exact': identities,
        'source_guard_counts': {k: len(plan[k]) for k in ['source_sha256', 'implementation_sha256', 'reference_sha256']},
        'additional_allocation_source_guards': extra, 'model_stat_guards_checked': len(plan['model_stat_guards']),
        'image_sha_checks': 16, 'npz_artifacts': artifacts, 'aggregate': list(aggregates.values()),
        'effective_selection_or_quota_changes': divergences,
        'quota_FP64_reference_differences': quota_floor_differences_from_FP64_reference,
        'actual_worker_exit': {'identity': identity, 'same_worker_still_alive': False,
            'proc_path_exists': proc.exists(), 'gpu_pid_absent': True, 'nvidia_smi': gpu.stdout,
            'root_exec': 34131, 'PTY_exit_code': 'Owned by root; not inferred by this audit'},
        'artifacts_sha256': {str(p): sha(p) for p in [HERE / 'run_plan.json', OUT / 'allocation_note.json',
            OUT / 'process_identity.json', OUT / 'loaded_runtime.json', OUT / 'per_image.jsonl', OUT / 'summary.json', Path(__file__)]},
        'GPU_started': False, 'model_loaded': False, 'GT_or_answers_read': False, 'benchmark_scores_computed': False,
        'claim_boundary': 'Terminal artifact verification only. Native five-crop GPU caches retained; no CPU projector substitute or new selection run. Fixed16 cannot establish full5000 accuracy or unpublished paper argv; ragged Eq6 cases remain unmeasured.'}
    (HERE / 'independent_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in {'npz_artifacts', 'artifacts_sha256', 'additional_allocation_source_guards'}}))


if __name__ == '__main__':
    main()
