#!/usr/bin/env python3
"""Independent CPU terminal audit; never load a model or start a container.

Saved raw bytes, selections and traces are exact checks. CPU cosine/entropy
mirrors are explicitly numerical consistency checks, not GPU bit reproductions.
The original GPU worker separately enforced equality to its native call.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stat(path):
    p = Path(path); s = p.stat()
    return dict(resolved=str(p.resolve()), bytes=s.st_size, device=s.st_dev,
                inode=s.st_ino, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def process_identity(pid):
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    return dict(pid=pid, start_ticks=int(fields[19]))


def bf16_values(np, bits):
    assert bits.dtype == np.uint16
    return (bits.astype(np.uint32) << 16).view(np.float32)


def bf16_round(np, value):
    fp32 = np.asarray(value, dtype=np.float32)
    raw = fp32.view(np.uint32)
    return ((raw + np.uint32(0x7fff) + ((raw >> 16) & 1)) >> 16).astype(np.uint16)


def meta(np, value, torch_dtype=None):
    return dict(dtype=torch_dtype or ('torch.' + str(value.dtype)), shape=list(value.shape),
                sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())


def embedding_rows(np, model_dir):
    """Map only the published tensor storage; indexing reads selected rows only."""
    root = Path(model_dir)
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    names = [k for k in index if k.endswith('embed_tokens.weight')]
    assert len(names) == 1, names
    name = names[0]; path = root / index[name]
    with path.open('rb') as f:
        header_length = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_length))
    item = header[name]
    assert item['dtype'] == 'BF16' and item['shape'][1] == 4096
    assert item['data_offsets'][1] - item['data_offsets'][0] == 2 * int(np.prod(item['shape']))
    tensor = np.memmap(path, mode='r', dtype='<u2',
                       offset=8 + header_length + item['data_offsets'][0], shape=tuple(item['shape']))
    return tensor, dict(path=str(path), tensor=name, dtype='BF16', shape=item['shape'],
                        full_payload_loaded_or_rehashed=False)


def numerical_mirrors(np, features_bits, sequence_bits, global_bits, global_sim, dense_sim, visual_sim):
    """Independent formulas on CPU. Never infer native IDs from CPU arithmetic."""
    visual = bf16_values(np, features_bits)
    sequence = bf16_values(np, sequence_bits)
    text_global = bf16_values(np, global_bits)
    L = sequence.shape[1]
    exact_mean = sequence.astype(np.float64).mean(axis=1)
    rounded_mean = bf16_round(np, exact_mean)
    # A CUDA BF16 mean accumulates FP32 then rounds to BF16. Bound an FP32
    # reduction/division, then require the exact real mean to lie in the saved
    # BF16 rounding cell enlarged by that bound. This permits no arbitrary cast.
    raw = global_bits.astype(np.int32)
    lower_raw = np.where((raw == 0) | (raw == 0x8000), 0x8001, np.where(raw > 0x8000, raw + 1, raw - 1))
    upper_raw = np.where(raw == 0x8000, 1, np.where(raw >= 0x8000, raw - 1, raw + 1))
    lower = bf16_values(np, lower_raw.astype(np.uint16)).astype(np.float64)
    upper = bf16_values(np, upper_raw.astype(np.uint16)).astype(np.float64)
    u = np.finfo(np.float32).eps / 2
    gamma = max(L - 1, 1) * u / (1 - max(L - 1, 1) * u)
    error = gamma * np.abs(sequence.astype(np.float64)).sum(axis=1) / L + u * np.abs(exact_mean)
    center = text_global.astype(np.float64)
    assert np.all(exact_mean >= (lower + center) / 2 - error)
    assert np.all(exact_mean <= (upper + center) / 2 + error)
    def norm32(value):
        return value / np.maximum(np.sqrt((value * value).sum(axis=-1, keepdims=True, dtype=np.float32)), np.float32(1e-8))
    image32 = norm32(visual)
    global_ref = -(image32 @ norm32(text_global).T)[None]
    dense_ref = -(image32 @ norm32(sequence[0]).T)[None, :, None, :]
    cross_delta = max(float(np.abs(global_ref - global_sim).max()), float(np.abs(dense_ref - dense_sim).max()))
    assert cross_delta <= 2e-5, ('CPU FP32 cross-cosine inconsistency', cross_delta)
    # Norm/division each round in BF16 before the FP32 GEMM. FP64 supplies
    # an independent norm reference; near BF16 rounding midpoints it need not
    # match a CUDA FP32 reduction bit-for-bit. The bound below admits one BF16
    # norm/division rounding difference; actual deltas are reported unhidden.
    norm64 = np.sqrt((visual.astype(np.float64) ** 2).sum(axis=-1, keepdims=True))
    norm_bf16 = bf16_values(np, bf16_round(np, norm64))
    normalized_bits = bf16_round(np, visual / np.maximum(norm_bf16, np.float32(1e-8)))
    normalized = bf16_values(np, normalized_bits)
    visual_ref = np.float32(.5) * (np.clip(normalized @ normalized.T, -1, 1) + np.float32(1))
    visual_delta = float(np.abs(visual_ref[None] - visual_sim).max())
    assert visual_delta <= .012, ('BF16-path CPU visual-cosine inconsistency', visual_delta)
    return dict(LLM_mean_FP64_RNE_bits_equal=bool(np.array_equal(rounded_mean, global_bits)),
                LLM_mean_FP32_reduction_BF16_rounding_interval_passed=True,
                LLM_mean_max_abs_delta_from_FP64=float(np.abs(center - exact_mean).max()),
                cross_cosine_CPU_FP32_max_abs_delta=cross_delta, cross_tolerance=2e-5,
                visual_cosine_CPU_BF16_round_reference_max_abs_delta=visual_delta, visual_tolerance=.012,
                visual_close_at_2e_6=visual_delta <= 2e-6,
                boundary='Independent CPU reference only; native GPU exact equalities remain separate worker gates. No CPU-selected IDs substitute for saved GPU IDs.')


def quantile_rule(np, dense, details):
    assert details['all_rows_same_length'] and details['source_members_and_order_retained']
    assert not details['padding_or_rowwise_reduction_used'] and len(details['rows']) == 1
    item = details['rows'][0]; L = dense.shape[-1]
    assert (item['batch'], item['segment'], item['L']) == (0, 0, L)
    old, selected, added = item['source_positions'], item['positions'], item['added_positions']
    assert len(old) == max(1, int(L * .2)) and len(set(old)) == len(old)
    assert selected == old + added and added == sorted(added) and len(set(selected)) == len(selected)
    assert all(0 <= k < L for k in selected) and not (set(old) & set(added))
    assert details['lengths'] == [len(selected)] and item['ties_at_threshold'] >= 0
    # Independent FP64 entropy reference on the exact saved FP32 dense input.
    # Its tolerance covers FP32 softmax/log/sum differences; membership/order
    # is also guaranteed by frozen inverse-proven source helper and native gate.
    x = dense[0, :, 0, :].astype(np.float64) * 100
    ex = np.exp(x - x.max(axis=0, keepdims=True)); probs = ex / ex.sum(axis=0, keepdims=True)
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=0)
    threshold = float(np.quantile(entropy, .2, method='linear'))
    saved = item['threshold']; tol = 3e-5
    assert abs(saved - threshold) <= tol
    excluded = sorted(set(range(L)) - set(selected))
    assert np.all(entropy[selected] <= saved + tol)
    if excluded:
        assert np.all(entropy[excluded] >= saved - tol)
    if len(old) < L:
        assert entropy[old].max() <= entropy[list(set(range(L)) - set(old))].min() + tol
    possible_ties = int((np.abs(entropy - saved) <= tol).sum())
    assert item['ties_at_threshold'] <= possible_ties
    return dict(L=L, source_members=len(old), threshold_members=len(selected), added_positions=added,
                stored_ties=item['ties_at_threshold'], CPU_FP64_threshold_delta=abs(saved - threshold),
                CPU_entropy_tolerance=tol, exact_GPU_tie_rule='Frozen source helper <= threshold; no CPU reclassification at FP32 tie boundaries')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=HERE)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--expected-allocation-sha256', required=True)
    parser.add_argument('--root-exit-proof', required=True, type=Path)
    args = parser.parse_args(); run = args.run_dir.resolve(); out = run / 'execution'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Use CUDA_VISIBLE_DEVICES= for this CPU checker'
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '2'
    import numpy as np
    assert not (run / 'failure.json').exists() and not (out / 'failure.json').exists(), 'Original terminal failure is retained; no successful numeric audit exists'
    plan_path = run / 'run_plan.json'; plan = json.loads(plan_path.read_text())
    assert sha(plan_path) == args.expected_plan_sha256
    state = json.loads((run / 'state.json').read_text()); summary = json.loads((out / 'summary.json').read_text())
    root_proof = json.loads(args.root_exit_proof.read_text())
    assert root_proof['root_owned_exec_actual_exit_code'] == 0
    assert root_proof['summary_sha256'] == sha(out / 'summary.json')
    allocation_path = run / 'root_allocation_note.json'; allocation = json.loads(allocation_path.read_text())
    identity = json.loads((out / 'identity.json').read_text()); runtime = json.loads((out / 'runtime.json').read_text())
    assert sha(allocation_path) == args.expected_allocation_sha256 == state['allocation_note_sha256'] == identity['allocation_note_sha256']
    assert identity['allocation'] == state['allocation'] == allocation
    assert state['status'] == 'complete' and state['exit_code'] == 0 and state['original_container_init_exited'] and state['GPU_pid_absent']
    assert summary['status'] in ('complete', 'complete_with_explicit_unmeasured')
    assert state['run_plan_sha256'] == summary['run_plan_sha256'] == identity['plan_sha256'] == allocation['run_plan_sha256'] == args.expected_plan_sha256
    assert state['summary_sha256'] == sha(out / 'summary.json')
    assert summary['n_questions'] == summary['native_get_calls'] == summary['native_anchors'] == summary['identity_controls'] == 16
    assert summary['planned_case_slots'] == 64 and summary['generated_answers'] == 0 and not summary['GT_read'] and not summary['benchmark_scores_computed']
    assert summary['per_question_sha256'] == sha(out / 'per_question.jsonl') and summary['runtime_sha256'] == sha(out / 'runtime.json')
    container = state['container_identity']; worker = container['host_init']
    assert identity['start_ticks'] == worker['start_ticks'] and container['container_id'] in container['host_cgroup']
    assert container['image'] == plan['container_image'] and allocation['GPU_slot'] == 'B'
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    gpu_pids = {int(line) for line in gpu.stdout.splitlines() if line.strip()}
    for previous in (worker, state['controller'], state['docker_client']):
        actual = process_identity(previous['pid'])
        assert actual is None or actual['start_ticks'] != previous['start_ticks'], previous
        assert previous['pid'] not in gpu_pids
    probe = subprocess.run(['docker', 'inspect', container['container_id']], capture_output=True, text=True)
    assert probe.returncode != 0 or not json.loads(probe.stdout)[0]['State']['Running']
    for group in ('source_sha256', 'implementation_sha256', 'reference_sha256'):
        assert all(sha(path) == expected for path, expected in plan[group].items()), group
    for path, guard in plan['model_stat_guards'].items():
        assert stat(path) == guard['current_full_stat']
    namespace_artifacts = []
    if 'model_bind_namespace' in plan:
        sys.path.insert(0, str(run))
        from namespace_guards import expected_model_stat, verify_mounts
        mapping = plan['model_bind_namespace']
        assert mapping['source_readonly'] and Path(mapping['source']).resolve() == Path(mapping['host_resolved_root'])
        probe_path = run / 'namespace_probe_report.json'
        probe_stats = json.loads(probe_path.read_text())
        assert sha(probe_path) == mapping['probe_report_sha256']
        assert not probe_stats['torch_imported'] and not probe_stats['model_loaded'] and not probe_stats['nvidia_device_exists']
        for path, guard in plan['model_stat_guards'].items():
            assert stat(path) == expected_model_stat(plan, path, 'host')
            assert probe_stats['model_stats'][path]['actual'] == expected_model_stat(plan, path, 'container')
            assert probe_stats['model_stats'][path]['host_expected'] == guard['current_full_stat']
        mount_path = run / 'container_mount_proof.json'; mounts = json.loads(mount_path.read_text())
        assert mounts['run_plan_sha256'] == args.expected_plan_sha256 and mounts['container_id'] == container['container_id']
        assert mounts['host_init'] == worker
        inspected = dict(Image=plan['container_image'], Mounts=[r['actual'] for r in mounts['bind_records']],
                         HostConfig=dict(Devices=mounts['actual_devices']))
        reproduced = verify_mounts(plan, inspected, gpu=True)
        assert all(mounts[k] == value for k, value in reproduced.items())
        namespace_artifacts = [probe_path, mount_path]
    for item in plan['panel']:
        assert sha(item['image_path']) == item['image_sha256']
    assert runtime['versions'] == plan['expected_versions'] and runtime['backend'] == runtime['vision_backend'] == 'flash_attention_2'
    assert runtime['model_dtype'] == runtime['visual_dtype'] == runtime['input_embedding_dtype'] == 'torch.bfloat16'
    assert runtime['default_dtype'] == 'torch.float32' and not runtime['autocast'] and not runtime['matmul_TF32'] and runtime['generate_calls'] == 0
    for path in runtime['source_modules'].values():
        assert sha(path) == plan['source_sha256'][path]
    modeling = runtime['modeling_source']; assert modeling['sha256'] == sha(modeling['path']) == plan['source_sha256'][plan['modeling_path']]
    # Verify the exact AST delta and unchanged division denominators without importing torch.
    sys.path.insert(0, str(run)); from formula_helpers import parts
    _, original, changed, _, _, proof = parts(plan['pruner_path'])
    assert proof == plan['source_AST_proof'] and proof['Eq10_inverse_restores_source'] and proof['observer_removed_restores_source']
    denominators = lambda node: [ast.dump(n.right, include_attributes=False) for n in ast.walk(node) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)]
    assert denominators(original) == denominators(changed)
    embedding, embedding_meta = embedding_rows(np, plan['recipe']['model_path'])
    rows = [json.loads(line) for line in (out / 'per_question.jsonl').read_text().splitlines()]
    assert len(rows) == 16
    artifacts = []; numerical = []; eq6_checks = []; changes = []; finite_arrays = identities = measured = skipped = 0
    aggregates = {v: dict(variant=v, measured=0, importance_changed=0, ordered_ids_changed=0, final_spatial_ids_changed=0) for v in plan['variants']}
    for number, (row, item) in enumerate(zip(rows, plan['panel'])):
        assert row['position'] == item['position'] and row['index'] == item['index'] and row['image_sha256'] == item['image_sha256']
        assert row['native_get_calls'] == 1 and row['native_pruner_anchor']
        path = out / ('question_%02d.npz' % number); assert str(path) == row['cache']['path'] and sha(path) == row['cache']['sha256']
        assert path.stat().st_size == row['cache']['bytes']; artifacts.append(row['cache'])
        assert [c['variant'] for c in row['cases']] == plan['variants']
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files) == set(row['tensors']) and len(z.files) == 13 + 4 * sum(c['status'] == 'measured' for c in row['cases'])
            for name in z.files:
                a = z[name]; expected = row['tensors'][name]
                assert list(a.shape) == expected['shape'] and hashlib.sha256(a.tobytes()).hexdigest() == expected['sha256']
                if expected['dtype'] == 'torch.bfloat16':
                    assert a.dtype == np.uint16 and np.isfinite(bf16_values(np, a)).all()
                else:
                    assert expected['dtype'] == 'torch.' + str(a.dtype) and np.isfinite(a).all()
                finite_arrays += 1
            features = z['visual_features_BF16_bits']; seq = z['text_sequence_BF16_bits']; mean = z['text_global_mean_BF16_bits']; L = row['actual_instruction_L']
            assert features.shape == (1024, 4096) and seq.shape == (1, L, 4096) and mean.shape == (1, 4096)
            assert z['instruction_ids'].shape == (1, L) and z['instruction_ids'].dtype == np.int64
            ids_for_embedding = z['instruction_ids']; assert np.all((ids_for_embedding >= 0) & (ids_for_embedding < embedding.shape[0]))
            assert np.array_equal(seq, embedding[ids_for_embedding]), 'Cached text must be exact native BF16 input-embedding rows'
            assert z['global_sim'].shape == (1, 1024, 1) and z['dense_sim'].shape == (1, 1024, 1, L) and z['visual_sim'].shape == (1, 1024, 1024)
            assert z['native_importance'].shape == (1, 1024) and z['native_ordered_ids'].shape == (1, 256) and z['native_spatial_ids'].shape == (256,)
            assert z['native_pruned_BF16_bits'].shape == (256, 4096) and np.array_equal(z['native_pruned_BF16_bits'], features[z['native_spatial_ids']])
            assert np.array_equal(z['image_grid_thw'], [[1, 64, 64]])
            assert meta(np, z['full_prompt_input_ids']) == item['processor_expected']['input_ids']
            assert meta(np, z['image_grid_thw']) == item['processor_expected']['image_grid_thw']
            numerical.append(dict(position=row['position'], **numerical_mirrors(np, features, seq, mean, z['global_sim'], z['dense_sim'], z['visual_sim'])))
            base_ids = z['native_ordered_ids']; base_spatial = z['native_spatial_ids']; base_weights = z['native_importance']; base_gain = z['source__gain_trace']
            assert np.array_equal(np.sort(base_ids[0]), base_spatial)
            for c in row['cases']:
                variant = c['variant']; aggregate = aggregates[variant]
                if c['status'] == 'unmeasured_ragged':
                    assert variant == 'eq6_quantile_membership_only' and len(set(c['text_filter']['lengths'])) > 1
                    assert not any(n.startswith(variant + '__') for n in z.files); skipped += 1; continue
                assert c['status'] == 'measured'
                weights, ids, spatial, gains = [z[variant + '__' + n] for n in ('importance', 'ordered_ids', 'spatial_ids', 'gain_trace')]
                assert weights.shape == (1, 1024) and weights.dtype == np.float32 and gains.shape == (256, 1, 1024) and gains.dtype == np.float32
                assert ids.shape == (1, 256) and ids.dtype == np.int64 and spatial.shape == (256,) and spatial.dtype == np.int64
                assert len(set(ids[0])) == 256 and np.all((ids >= 0) & (ids < 1024)) and np.array_equal(spatial, np.sort(ids[0]))
                assert np.array_equal(np.argmax(gains, axis=2).T, ids)
                for t in range(256): assert np.all(gains[t, 0, ids[0, :t]] == np.float32(-1e9))
                importance_same = np.array_equal(weights, base_weights); order_same = np.array_equal(ids, base_ids); spatial_same = np.array_equal(spatial, base_spatial)
                assert c['importance_exact_equal'] == importance_same and c['importance_max_abs_delta'] == float(np.abs(weights - base_weights).max())
                assert c['ordered_ids_equal'] == order_same and c['final_spatial_ids_equal'] == spatial_same
                assert c['ordered_ids'] == ids[0].tolist() and c['spatial_ids'] == spatial.tolist() and c['actual_retained'] == 256 and c['traced_matches_native_greedy']
                assert c['final_spatial_features'] == meta(np, features[spatial], 'torch.bfloat16')
                added = sorted(set(spatial.tolist()) - set(base_spatial.tolist())); removed = sorted(set(base_spatial.tolist()) - set(spatial.tolist()))
                assert c['added'] == added and c['removed'] == removed
                first = next((t for t in range(256) if ids[0, t] != base_ids[0, t]), None)
                divergence = None
                if first is not None:
                    picks = [int(base_ids[0, first]), int(ids[0, first])]
                    divergence = dict(step=first, same_prefix_before_step=np.array_equal(ids[:, :first], base_ids[:, :first]), baseline_variant_picks=picks,
                                      baseline_gains=[float(base_gain[first, 0, k]) for k in picks], variant_gains=[float(gains[first, 0, k]) for k in picks])
                assert c['first_divergence'] == divergence
                if variant == 'source': assert importance_same and order_same and spatial_same
                if variant == 'source_members_order_identity':
                    assert importance_same and order_same and spatial_same and np.array_equal(gains, base_gain); identities += 1
                if variant == 'eq6_quantile_membership_only': eq6_checks.append(dict(position=row['position'], **quantile_rule(np, z['dense_sim'], c['text_filter'])))
                if not order_same or not spatial_same: changes.append(dict(position=row['position'], index=row['index'], variant=variant, ordered_changed=not order_same, actual_spatial_set_changed=not spatial_same, added=added, removed=removed, first_divergence=divergence))
                aggregate['measured'] += 1; aggregate['importance_changed'] += not importance_same
                aggregate['ordered_ids_changed'] += not order_same; aggregate['final_spatial_ids_changed'] += not spatial_same; measured += 1
        print(json.dumps(dict(independently_checked_questions=number + 1, measured_cases=measured)), flush=True)
    assert measured + skipped == 64 and identities == 16
    assert summary['measured_cases'] == measured and summary['unmeasured_cases'] == skipped and (summary['status'] == 'complete') == (skipped == 0)
    assert summary['aggregate'] == list(aggregates.values()) and summary['cache_bytes'] == sum(a['bytes'] for a in artifacts)
    assert 'torch' not in sys.modules, 'No tensor runtime, model or CUDA is imported by this auditor'
    report = dict(success=True, checked_utc=datetime.now(timezone.utc).isoformat(), n_questions=16, unique_images=len({r['image_path'] for r in plan['panel']}),
                  planned_case_slots=64, measured_cases=measured, unmeasured_cases=skipped, finite_arrays_checked=finite_arrays,
                  exact_native_source_anchors=16, exact_identity_controls=identities, embedding_rows_exact=16,
                  embedding_reference=embedding_meta, numerical_consistency=numerical, Eq6_checks=eq6_checks,
                  Eq10_AST_denominators_unchanged=True, aggregate=list(aggregates.values()), selection_changes=changes,
                  source_guard_counts={k: len(plan[k]) for k in ('source_sha256', 'implementation_sha256', 'reference_sha256')},
                  model_stat_guards_checked=len(plan['model_stat_guards']), image_SHA_checks=16, npz_artifacts=artifacts,
                  actual_exit=dict(container=container, controller=state['controller'], docker_client=state['docker_client'], original_processes_exited=True, GPU_pids=gpu.stdout),
                  artifacts_sha256={str(p): sha(p) for p in [plan_path, allocation_path, args.root_exit_proof, run / 'state.json', out / 'identity.json', out / 'runtime.json', out / 'summary.json', out / 'per_question.jsonl', Path(__file__), *namespace_artifacts]},
                  GPU_started=False, container_started=False, model_loaded=False, GT_or_answers_read=False, benchmark_scores_computed=False,
                  claim_boundary='Exact artifact/embedding/selection/trace checks plus explicitly bounded CPU formula references. The native GPU worker enforced separate bit-exact source+identity anchors. Fixed16 cannot establish full benchmark accuracy, unpublished paper argv or a validated repair.')
    destination = run / 'independent_verification.json'
    assert not destination.exists(), 'Do not overwrite an earlier terminal report'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(success=True, report=str(destination), sha256=sha(destination), measured_cases=measured, aggregate=report['aggregate'])))


if __name__ == '__main__':
    main()
