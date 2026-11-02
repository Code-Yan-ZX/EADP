#!/usr/bin/env python3
"""CPU-only precision explanation from the fixed, independently verified 16 caches."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '2'
    import numpy as np
    verified = json.loads((HERE / 'independent_verification.json').read_text())
    assert verified['success'] and verified['measured_cases'] == 64 and verified['unmeasured_cases'] == 0
    records = [json.loads(line) for line in (HERE / 'execution/per_question.jsonl').read_text().splitlines()]
    assert len(records) == 16
    results = []
    for row in records:
        case = next(c for c in row['cases'] if c['variant'] == 'eq6_quantile_membership_only')
        info = case['text_filter']['rows'][0]
        old = np.array(info['source_positions'], dtype=np.int64)
        selected = np.array(info['positions'], dtype=np.int64)
        added = info['added_positions']
        assert selected.tolist() == old.tolist() + added
        with np.load(row['cache']['path'], allow_pickle=False) as z:
            dense = z['dense_sim'][0, :, 0, :]; global_sim = z['global_sim'][0, :, 0]
            def reference(dtype, positions):
                d = dense.astype(dtype); logits = d * dtype(100.)
                probabilities = np.exp(logits - logits.max(axis=0, keepdims=True))
                probabilities /= probabilities.sum(axis=0, keepdims=True, dtype=dtype)
                entropy = -(probabilities * np.log(probabilities + dtype(1e-12))).sum(axis=0, dtype=dtype)
                local_logits = -entropy[positions] / dtype(.010001)
                weights = np.exp(local_logits - local_logits.max())
                weights /= weights.sum(dtype=dtype)
                local = (d[:, positions] * weights).sum(axis=-1, dtype=dtype)
                fusion = dtype(.5) * global_sim.astype(dtype) + dtype(.5) * local
                importance = (fusion - fusion.min() + dtype(1e-6)) / (fusion.max() - fusion.min() + dtype(1e-6))
                return entropy, weights, local, fusion, importance
            H64, w064, local064, fusion064, _ = reference(np.float64, old)
            _, w164, local164, fusion164, _ = reference(np.float64, selected)
            H32, w032, local032, fusion032, imp032 = reference(np.float32, old)
            _, w132, local132, fusion132, imp132 = reference(np.float32, selected)
            native_importance_equal = np.array_equal(z['source__importance'], z['eq6_quantile_membership_only__importance'])
            native_ordered_equal = np.array_equal(z['source__ordered_ids'], z['eq6_quantile_membership_only__ordered_ids'])
            native_spatial_equal = np.array_equal(z['source__spatial_ids'], z['eq6_quantile_membership_only__spatial_ids'])
            assert native_importance_equal and native_ordered_equal and native_spatial_equal
            results.append(dict(position=row['position'], canonical_index=row['index'], L=dense.shape[-1],
                source_members=old.tolist(), threshold_members=selected.tolist(), added_positions=added,
                added_entropy_gap_from_source_min_FP64=[float(H64[k] - H64[old].min()) for k in added],
                added_weight_CPU_FP64=w164[len(old):].tolist(), added_weight_CPU_FP32=w132[len(old):].tolist(),
                source_prefix_weights_CPU_FP32_exact=np.array_equal(w032, w132[:len(old)]),
                added_FP32_weights_positive_normal=bool(np.all(w132[len(old):] >= np.finfo(np.float32).tiny)) if added else None,
                added_weight_sum_CPU_FP64=float(w164[len(old):].sum()),
                local_CPU_FP64_max_abs_delta=float(np.abs(local164 - local064).max()),
                fusion_CPU_FP64_max_abs_delta=float(np.abs(fusion164 - fusion064).max()),
                local_CPU_FP32_changed_positions=int((local132 != local032).sum()),
                local_CPU_FP32_max_abs_delta=float(np.abs(local132 - local032).max()),
                fusion_CPU_FP32_changed_positions=int((fusion132 != fusion032).sum()),
                fusion_CPU_FP32_max_abs_delta=float(np.abs(fusion132 - fusion032).max()),
                pre_smoothing_importance_CPU_FP32_changed_positions=int((imp132 != imp032).sum()),
                pre_smoothing_importance_CPU_FP32_max_abs_delta=float(np.abs(imp132 - imp032).max()),
                native_GPU_final_importance_exact=native_importance_equal,
                native_GPU_ordered256_exact=native_ordered_equal, native_GPU_final_spatial256_exact=native_spatial_equal))
    appended = [r for r in results if r['added_positions']]
    assert len(appended) == 10 and all(len(r['added_positions']) == 1 for r in appended)
    assert all(r['source_prefix_weights_CPU_FP32_exact'] for r in results)
    assert all(r['added_FP32_weights_positive_normal'] for r in appended)
    assert 'torch' not in sys.modules
    report = dict(success=True, checked_utc=datetime.now(timezone.utc).isoformat(), fixed_questions=16,
        L_range=[min(r['L'] for r in results), max(r['L'] for r in results)],
        questions_with_actual_added_member=10, questions_without_membership_change=6,
        native_GPU_importance_and_ordered_and_final_spatial_unchanged=16,
        added_weight_CPU_FP64_range=[min(r['added_weight_CPU_FP64'][0] for r in appended), max(r['added_weight_CPU_FP64'][0] for r in appended)],
        added_weight_CPU_FP32_range=[min(r['added_weight_CPU_FP32'][0] for r in appended), max(r['added_weight_CPU_FP32'][0] for r in appended)],
        all_added_weights_representable_normal_FP32=True,
        all_source_prefix_weights_CPU_FP32_exact=True,
        CPU_FP64_local_delta_max=max(r['local_CPU_FP64_max_abs_delta'] for r in results),
        CPU_FP32_local_delta_max=max(r['local_CPU_FP32_max_abs_delta'] for r in results),
        CPU_FP32_fusion_delta_max=max(r['fusion_CPU_FP32_max_abs_delta'] for r in results),
        CPU_FP32_local_change_questions=sum(r['local_CPU_FP32_changed_positions'] > 0 for r in results),
        CPU_FP32_pre_smoothing_change_questions=sum(r['pre_smoothing_importance_CPU_FP32_changed_positions'] > 0 for r in results),
        formula=dict(negative_entropy_temperature=.01, denominator_with_source_epsilon=.010001,
                     cross_similarity_input='Exact cached native GPU FP32 dense/global, not recomputed features',
                     convention='Source captured member order; two independent NumPy CPU precision references'),
        records=results, input_sha256={str(p): sha(p) for p in [HERE/'independent_verification.json', HERE/'execution/per_question.jsonl', HERE/'execution/summary.json', Path(__file__)]},
        GPU_started=False, model_loaded=False, torch_imported=False, GT_or_answers_read=False,
        limitations=['CPU softmax/log/sum are numerical mirrors, not saved GPU intermediate weights or GPU reduction replay.',
                     'All10 added weights are positive normal FP32 in the CPU mirror; no underflow/NaN explanation is claimed.',
                     'The very small added mass and observed exact native final importance support a finite-precision-insensitivity explanation on this panel; the first exact cancellation point inside GPU arithmetic was not recorded.',
                     'Actual native final importance/order/set equality is established by the independent terminal artifact audit, separate from these CPU mirrors.',
                     'Fixed16 does not establish the whole5349 benchmark, accuracy recovery, paper argv, or a repair; no full variant is automatically authorized.'])
    target = HERE / 'eq6_weight_precision_boundary.json'; assert not target.exists()
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','input_sha256','limitations')}))
    print('report_sha256',sha(target))


if __name__ == '__main__': main()
